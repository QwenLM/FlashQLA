import ast
import copy
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_bwd.py"
)


def _tree(source=None):
    return ast.parse((source or SOURCE).read_text())


def _unparse(node):
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        if hasattr(child, "ctx"):
            child.ctx = ast.Load()
    return ast.dump(node)


def _expression(source):
    return _unparse(ast.parse(source, mode="eval").body)


def allocation_targets(source=None):
    tree = _tree(source)
    assignments = {}
    alloc_tmem_targets = []
    alloc_fragment_targets = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        assignments[target.id] = node.value
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "T"
        ):
            entry = (node.lineno, target.id)
            if node.value.func.attr == "alloc_tmem":
                alloc_tmem_targets.append(entry)
            elif node.value.func.attr == "alloc_fragment":
                alloc_fragment_targets.append(entry)

    alloc_tmem_targets = [name for _, name in sorted(alloc_tmem_targets)]
    alloc_fragment_targets = [name for _, name in sorted(alloc_fragment_targets)]
    return assignments, alloc_tmem_targets, alloc_fragment_targets


def dataflow_events(source=None):
    events = []
    for node in ast.walk(_tree(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "T"
            and node.func.attr
            in {
                "copy",
                "reduce_sum",
                "barrier_arrive",
                "barrier_wait",
                "tcgen05_gemm",
            }
        ):
            name = node.func.attr
            args = tuple(_unparse(arg) for arg in node.args)
            keywords = {
                keyword.arg: _unparse(keyword.value) for keyword in node.keywords
            }
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Mult):
            name = "multiply"
            args = (_unparse(node.target), _unparse(node.value))
            keywords = {}
        else:
            continue
        events.append(
            {
                "line": node.lineno,
                "name": name,
                "args": args,
                "keywords": keywords,
            }
        )
    return sorted(events, key=lambda event: event["line"])


def _is_t_call(node, names):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "T"
        and node.func.attr in names
    )


def _t_call_event(call):
    return {
        "line": call.lineno,
        "name": call.func.attr,
        "args": tuple(_unparse(arg) for arg in call.args),
        "arg_nodes": tuple(call.args),
        "keywords": {
            keyword.arg: _unparse(keyword.value) for keyword in call.keywords
        },
        "keyword_nodes": {
            keyword.arg: keyword.value for keyword in call.keywords
        },
    }


def _t_call_events(statements, names):
    events = []
    for statement in statements:
        for node in ast.walk(statement):
            if _is_t_call(node, names):
                events.append(_t_call_event(node))
    return sorted(events, key=lambda event: event["line"])


def _event_arg_name(event, index=0):
    if len(event["arg_nodes"]) <= index:
        return None
    arg = event["arg_nodes"][index]
    return arg.id if isinstance(arg, ast.Name) else None


def _event_mbar_name(event):
    mbar = event["keyword_nodes"].get("mbar")
    return mbar.id if isinstance(mbar, ast.Name) else None


def _is_serial_num_iters_loop(node):
    return (
        isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and isinstance(node.iter.func.value, ast.Name)
        and node.iter.func.value.id == "T"
        and node.iter.func.attr == "serial"
        and len(node.iter.args) == 1
        and not node.iter.keywords
        and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "num_iters"
    )


def producer_iteration_events(source=None):
    candidates = []
    for node in ast.walk(_tree(source)):
        if not _is_serial_num_iters_loop(node):
            continue
        events = _t_call_events(
            node.body, {"barrier_arrive", "barrier_wait", "tcgen05_gemm"}
        )
        wait_names = {
            _event_arg_name(event)
            for event in events
            if event["name"] == "barrier_wait"
        }
        if (
            {"bar_08", "bar_09", "bar_10", "bar_11"} <= wait_names
            and any(event["name"] == "tcgen05_gemm" for event in events)
        ):
            candidates.append(events)

    assert len(candidates) == 1, (
        "expected one producer serial(num_iters) loop, found {}".format(
            len(candidates)
        )
    )
    return candidates[0]


def _producer_wait_line(events, barrier):
    lines = [
        event["line"]
        for event in events
        if event["name"] == "barrier_wait"
        and _event_arg_name(event) == barrier
    ]
    assert len(lines) == 1, "expected one producer wait on {}".format(barrier)
    return lines[0]


def _mbar_events(events, name):
    return [
        event
        for event in events
        if event["name"] == "tcgen05_gemm"
        and _event_mbar_name(event) == name
    ]


def producer_stage08_state_v_first_gemm_branches(source=None):
    producer_lines = {
        event["line"] for event in producer_iteration_events(source)
    }
    candidates = []
    for node in ast.walk(_tree(source)):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "state_v_first"
        ):
            continue

        branches = [
            _t_call_events(statements, {"tcgen05_gemm"})
            for statements in (node.body, node.orelse)
        ]
        stage08_events = [
            event
            for branch in branches
            for event in _mbar_events(branch, "tcbar_08_L")
            + _mbar_events(branch, "tcbar_08_R")
        ]
        if stage08_events and all(
            event["line"] in producer_lines for event in stage08_events
        ):
            candidates.append(tuple(branches))

    assert len(candidates) == 1, (
        "expected one state_v_first producer stage-08 branch, found {}".format(
            len(candidates)
        )
    )
    return candidates[0]


def event_lines(events, name, args, keywords=None, required=True):
    expected_args = tuple(_expression(arg) for arg in args)
    expected_keywords = {
        key: _expression(value) for key, value in (keywords or {}).items()
    }
    lines = [
        event["line"]
        for event in events
        if event["name"] == name
        and event["args"] == expected_args
        and event["keywords"] == expected_keywords
    ]
    if required:
        assert lines, "missing T.{}({}, {})".format(name, args, keywords or {})
    return lines


def parallel_multiply_lines(target, value, parallel_args, source=None):
    expected_target = _expression(target)
    expected_value = _expression(value)
    expected_parallel_args = tuple(_expression(arg) for arg in parallel_args)
    lines = []

    for node in ast.walk(_tree(source)):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
            continue
        parallel = node.iter
        if not (
            isinstance(parallel.func, ast.Attribute)
            and isinstance(parallel.func.value, ast.Name)
            and parallel.func.value.id == "T"
            and parallel.func.attr == "Parallel"
            and tuple(_unparse(arg) for arg in parallel.args)
            == expected_parallel_args
        ):
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.AugAssign)
                and isinstance(statement.op, ast.Mult)
                and _unparse(statement.target) == expected_target
                and _unparse(statement.value) == expected_value
            ):
                lines.append(statement.lineno)

    return sorted(lines)


def _root_buffer_name(node):
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _assignment_targets(target):
    if isinstance(target, (ast.List, ast.Tuple)):
        targets = []
        for element in target.elts:
            targets.extend(_assignment_targets(element))
        return targets
    return [target]


def subscript_accesses_between(buffer_name, start_line, end_line, source=None):
    accesses = []
    for node in ast.walk(_tree(source)):
        if not (
            isinstance(node, ast.Subscript)
            and start_line < node.lineno < end_line
            and _root_buffer_name(node) == buffer_name
        ):
            continue
        accesses.append(
            {
                "line": node.lineno,
                "expression": _unparse(node),
                "context": type(node.ctx).__name__,
            }
        )
    return sorted(accesses, key=lambda access: access["line"])


def buffer_mutations_between(buffer_name, start_line, end_line, source=None):
    mutations = []
    for node in ast.walk(_tree(source)):
        line = getattr(node, "lineno", 0)
        if not start_line < line <= end_line:
            continue

        kind = None
        targets = []
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "T"
        ):
            if node.func.attr == "copy" and len(node.args) >= 2:
                kind = "T.copy"
                targets = [node.args[1]]
            elif node.func.attr in {"clear", "fill"} and node.args:
                kind = "T.{}".format(node.func.attr)
                targets = [node.args[0]]
        elif isinstance(node, ast.Assign):
            kind = "Assign"
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            kind = "AnnAssign"
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            kind = "AugAssign"
            targets = [node.target]

        for target in targets:
            for assignment_target in _assignment_targets(target):
                if _root_buffer_name(assignment_target) != buffer_name:
                    continue
                mutations.append(
                    {
                        "line": line,
                        "kind": kind,
                        "target": _unparse(assignment_target),
                    }
                )

    return sorted(
        mutations,
        key=lambda mutation: (
            mutation["line"],
            mutation["kind"],
            mutation["target"],
        ),
    )


def test_dq_tmem_reuses_u_tmem_allocation():
    assignments, _, alloc_fragment_targets = allocation_targets()

    dq_assignment = assignments["dq_tmem"]
    dq_fragment_assignment = assignments["dq_fragment"]
    assert isinstance(dq_assignment, ast.Name), ast.dump(dq_assignment)
    assert isinstance(dq_fragment_assignment, ast.Name), ast.dump(
        dq_fragment_assignment
    )
    assert dq_assignment.id == "u_tmem"
    assert dq_fragment_assignment.id == "u_fragment"
    assert "dq_fragment" not in alloc_fragment_targets


def _assert_single_arrival_tmem_barrier(assignments, name):
    assert name in assignments
    allocation = assignments[name]
    assert isinstance(allocation, ast.Call), ast.dump(allocation)
    assert (
        isinstance(allocation.func, ast.Attribute)
        and isinstance(allocation.func.value, ast.Name)
        and allocation.func.value.id == "T"
        and allocation.func.attr == "alloc_barrier"
    ), ast.dump(allocation)
    assert not allocation.args
    assert [
        (keyword.arg, _unparse(keyword.value)) for keyword in allocation.keywords
    ] == [("arrive_count", "1")]


def test_full_dq_barriers_have_one_arrival_each():
    assignments, _, _ = allocation_targets()

    for name in ("tcbar_08", "tcbar_10"):
        _assert_single_arrival_tmem_barrier(assignments, name)
    assert not {
        name
        for name in ("tcbar_08_L", "tcbar_08_R", "tcbar_10_L", "tcbar_10_R")
        if name in assignments
    }


def test_producer_stage08_uses_full_h_shared_and_dq_tmem():
    events = producer_iteration_events()
    lines = event_lines(
        events,
        "tcgen05_gemm",
        ["do_shared", "h_shared", "dq_tmem"],
        {
            "transpose_B": "not state_v_first",
            "clear_accum": "True",
            "mbar": "tcbar_08",
            "use_2cta": "False",
        },
    )
    assert len(lines) == 1
    signals = _mbar_events(events, "tcbar_08")
    assert len(signals) == 1
    assert signals[0]["line"] == lines[0]


def test_producer_stage10_uses_full_tmp_shared_2_2_without_clearing():
    events = producer_iteration_events()
    assert len(
        event_lines(
            events,
            "tcgen05_gemm",
            ["tmp_shared_1_1", "tmp_shared_2_2", "dq_tmem"],
            {
                "clear_accum": "False",
                "mbar": "tcbar_10",
                "use_2cta": "False",
            },
        )
    ) == 1


def test_producer_full_dq_gemms_have_single_mbars_and_stage_bounds():
    events = producer_iteration_events()
    wait_08 = _producer_wait_line(events, "bar_08")
    wait_09 = _producer_wait_line(events, "bar_09")
    wait_10 = _producer_wait_line(events, "bar_10")
    wait_11 = _producer_wait_line(events, "bar_11")

    stage08 = _mbar_events(events, "tcbar_08")
    stage10 = _mbar_events(events, "tcbar_10")
    assert len(stage08) == len(stage10) == 1
    assert wait_08 < stage08[0]["line"] < wait_09
    assert wait_10 < stage10[0]["line"] < wait_11


def test_mask_tmem_has_explicit_tcgen05_layout_e():
    tree = _tree()
    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "tilelang_fused_chunk_gdr_bwd"
    )
    factory_assignments = {
        node.targets[0].id: node
        for node in factory.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }

    assert "mask_tmem_layout" in factory_assignments
    layout_assignment = factory_assignments["mask_tmem_layout"]
    assert factory_assignments["block_S"].lineno < layout_assignment.lineno
    kernel = next(
        node
        for node in factory.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "tilelang_fused_chunk_gdr_bwd_kernel"
    )
    assert layout_assignment.lineno < kernel.lineno

    layout_call = layout_assignment.value
    assert isinstance(layout_call, ast.Call), ast.dump(layout_call)
    assert (
        isinstance(layout_call.func, ast.Attribute)
        and layout_call.func.attr == "Layout"
        and isinstance(layout_call.func.value, ast.Attribute)
        and layout_call.func.value.attr == "layout"
        and isinstance(layout_call.func.value.value, ast.Name)
        and layout_call.func.value.value.id == "tilelang"
    ), "mask_tmem_layout must be an ordinary tilelang.layout.Layout"
    assert len(layout_call.args) == 2 and not layout_call.keywords
    shape, forward = layout_call.args
    assert isinstance(shape, ast.List)
    assert [element.id for element in shape.elts] == ["block_S", "block_S"]
    assert isinstance(forward, ast.Lambda)
    assert [argument.arg for argument in forward.args.args] == ["i", "j"]
    expected_forward = ast.parse(
        "[i + (j // 32) * 64, j % 32]", mode="eval"
    ).body
    assert ast.dump(forward.body) == ast.dump(expected_forward)

    annotation_calls = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "T"
        and node.func.attr == "annotate_layout"
    ]
    assert len(annotation_calls) == 1
    annotation = annotation_calls[0]
    assert len(annotation.args) == 1 and not annotation.keywords
    assert isinstance(annotation.args[0], ast.Dict)
    layout_entries = {
        key.id: value
        for key, value in zip(annotation.args[0].keys, annotation.args[0].values)
        if isinstance(key, ast.Name)
    }
    assert isinstance(layout_entries.get("mask_tmem"), ast.Name)
    assert layout_entries["mask_tmem"].id == "mask_tmem_layout"

    assignments, alloc_tmem_targets, _ = allocation_targets()
    mask_storage = assignments["mask_tmem"]
    assert "mask_tmem" in alloc_tmem_targets
    assert isinstance(mask_storage, ast.Call)
    assert isinstance(mask_storage.func, ast.Attribute)
    assert mask_storage.func.attr == "alloc_tmem"


def test_tcgen05_layout_e_is_a_64x64_to_128x32_bijection():
    physical_coordinates = {
        (i + (j // 32) * 64, j % 32)
        for i in range(64)
        for j in range(64)
    }
    expected_coordinates = {
        (row, column) for row in range(128) for column in range(32)
    }

    assert len(physical_coordinates) == 64 * 64
    assert physical_coordinates == expected_coordinates


def test_consumer_a_spill_fragments_move_to_mask_tmem():
    assignments, alloc_tmem_targets, alloc_fragment_targets = allocation_targets()

    assert "mask_tmem" in alloc_tmem_targets
    assert len(alloc_tmem_targets) == 10
    assert "mask_fragment" not in alloc_fragment_targets
    assert "odot_fragment_2" not in alloc_fragment_targets

    expected_consumer_a = [
        "p_fragment",
        "a_fragment",
        "dp_fragment",
        "da_fragment",
        "u_fragment",
        "db_fragment",
        "dg_fragment_2",
    ]
    assert "dq_fragment" not in alloc_fragment_targets
    consumer_a_start = alloc_fragment_targets.index("dg_last_local_1") + 1
    consumer_a_end = consumer_a_start + len(expected_consumer_a)
    assert alloc_fragment_targets[consumer_a_start:consumer_a_end] == expected_consumer_a
    assert alloc_fragment_targets[consumer_a_end] == "dh_fragment_L"

    mask_allocation = assignments["mask_tmem"]
    assert isinstance(mask_allocation, ast.Call), ast.dump(mask_allocation)
    assert len(mask_allocation.args) == 1
    expected_shape = ast.parse("(block_S, block_S)", mode="eval").body
    assert ast.dump(mask_allocation.args[0]) == ast.dump(expected_shape)


def test_stage08_q_snapshot_uses_only_shared_storage():
    events = dataflow_events()

    a_to_mask = event_lines(
        events, "copy", ["a_fragment", "mask_tmem"], required=False
    )
    mask_to_a = event_lines(
        events, "copy", ["mask_tmem", "a_fragment"], required=False
    )
    assert len(a_to_mask) == 1, "a_fragment -> mask_tmem must only store stage-00 mask"
    assert len(mask_to_a) == 1, "mask_tmem -> a_fragment must only load stage-07 mask"

    q_left = event_lines(
        events, "copy", ["q_shared[:, :DK // 2]", "a_fragment"]
    )[0]
    q_left_shared = event_lines(
        events,
        "copy",
        ["a_fragment", "tmp_shared_2_1[:, :DK // 2]"],
    )[0]
    q_right = event_lines(
        events, "copy", ["q_shared[:, DK // 2:]", "p_fragment"]
    )[0]
    q_right_shared = event_lines(
        events,
        "copy",
        ["p_fragment", "tmp_shared_2_1[:, DK // 2:]"],
    )[0]
    bar_09 = next(
        line
        for line in event_lines(events, "barrier_arrive", ["bar_09"])
        if line > q_right_shared
    )
    stage08_copies = [
        event["args"]
        for event in events
        if event["name"] == "copy" and q_left <= event["line"] < bar_09
    ]

    assert stage08_copies == [
        (_expression("q_shared[:, :DK // 2]"), _expression("a_fragment")),
        (
            _expression("a_fragment"),
            _expression("tmp_shared_2_1[:, :DK // 2]"),
        ),
        (_expression("q_shared[:, DK // 2:]"), _expression("p_fragment")),
        (
            _expression("p_fragment"),
            _expression("tmp_shared_2_1[:, DK // 2:]"),
        ),
    ]
    assert q_left < q_left_shared < q_right < q_right_shared < bar_09


def _consumer_a_dq_stage_bounds(events):
    snapshot = event_lines(
        events,
        "copy",
        ["p_fragment", "tmp_shared_2_1[:, DK // 2:]"],
    )[0]
    tcbar_08_wait = next(
        line
        for line in event_lines(
            events, "barrier_wait", ["tcbar_08", "(i_s + 0) % 2"]
        )
        if line > snapshot
    )
    bar_09_arrive = next(
        line
        for line in event_lines(events, "barrier_arrive", ["bar_09"])
        if line > tcbar_08_wait
    )
    wait_09 = next(
        line
        for line in event_lines(
            events, "barrier_wait", ["bar_09", "(i_s + 0) % 2"]
        )
        if line > bar_09_arrive
    )
    bar_10_arrive = next(
        line
        for line in event_lines(events, "barrier_arrive", ["bar_10"])
        if line > wait_09
    )
    wait_10 = next(
        line
        for line in event_lines(
            events, "barrier_wait", ["bar_10", "(i_s + 0) % 2"]
        )
        if line > bar_10_arrive
    )
    tcbar_10_wait = next(
        line
        for line in event_lines(
            events, "barrier_wait", ["tcbar_10", "(i_s + 0) % 2"]
        )
        if line > wait_10
    )
    bar_11_arrive = next(
        line
        for line in event_lines(events, "barrier_arrive", ["bar_11"])
        if line > tcbar_10_wait
    )
    assert snapshot < tcbar_08_wait < bar_09_arrive < wait_09 < bar_10_arrive
    assert wait_10 < tcbar_10_wait < bar_11_arrive
    return wait_09, bar_10_arrive, tcbar_10_wait, bar_11_arrive


def test_consumer_a_waits_once_for_full_dq_tmem_barriers():
    events = dataflow_events()
    snapshot = event_lines(
        events,
        "copy",
        ["p_fragment", "tmp_shared_2_1[:, DK // 2:]"],
    )[0]
    tcbar_08_waits = [
        line
        for line in event_lines(
            events, "barrier_wait", ["tcbar_08", "(i_s + 0) % 2"]
        )
        if line > snapshot
    ]
    _, _, tcbar_10_wait, _ = _consumer_a_dq_stage_bounds(events)
    assert len(tcbar_08_waits) == 1
    assert tcbar_10_wait > tcbar_08_waits[0]


def test_stage09_full_dq_streams_through_reused_u_fragment():
    events = dataflow_events()
    wait_09, bar_10, _, _ = _consumer_a_dq_stage_bounds(events)

    def in_stage09(lines):
        return [line for line in lines if wait_09 < line < bar_10]

    full_load = in_stage09(
        event_lines(events, "copy", ["dq_tmem", "dq_fragment"])
    )
    full_store = in_stage09(
        event_lines(events, "copy", ["dq_fragment", "dq_tmem"])
    )
    g_scales = in_stage09(
        parallel_multiply_lines(
            "dq_fragment[j_s, j_k]",
            "g_exp_shared[j_s]",
            ["block_S", "DK"],
        )
    )
    dot = in_stage09(
        parallel_multiply_lines(
            "dq_fragment[j_s, j_k]",
            "tmp_shared_2_1[j_s, j_k]",
            ["block_S", "DK"],
        )
    )
    reductions = in_stage09(
        event_lines(
            events,
            "reduce_sum",
            ["dq_fragment", "dg_fragment_2"],
            {"dim": "1", "clear": "False"},
        )
    )

    assert all(
        len(lines) == 1
        for lines in (full_load, full_store, g_scales, dot, reductions)
    )
    assert (
        full_load[0]
        < g_scales[0]
        < full_store[0]
        < dot[0]
        < reductions[0]
        < bar_10
    )


def test_stage10_publishes_full_dq_after_tcbar10():
    events = dataflow_events()
    _, _, tcbar_10_wait, bar_11 = _consumer_a_dq_stage_bounds(events)

    def in_stage10(lines):
        return [line for line in lines if tcbar_10_wait < line < bar_11]

    full_load = in_stage10(
        event_lines(events, "copy", ["dq_tmem", "dq_fragment"])
    )
    full_store = in_stage10(
        event_lines(events, "copy", ["dq_fragment", "dqkv_shared"])
    )

    assert len(full_load) == len(full_store) == 1
    assert tcbar_10_wait < full_load[0] < full_store[0] < bar_11


def test_full_dq_is_never_sliced_as_a_tmem_operand():
    partial_views = [
        _unparse(node)
        for node in ast.walk(_tree())
        if isinstance(node, ast.Subscript) and _root_buffer_name(node) == "dq_tmem"
    ]
    assert not partial_views, partial_views


def test_shared_q_snapshot_keeps_stage12_and_stage14_gemm_signatures():
    events = dataflow_events()
    signatures = [
        (
            ["tmp_shared_1_1", "tmp_shared_2_1", "dk_tmem"],
            {
                "transpose_A": "True",
                "clear_accum": "False",
                "mbar": "tcbar_12",
                "use_2cta": "False",
            },
        ),
        (
            [
                "tmp_shared_2_3",
                "tmp_shared_2_1[:, :DK // 2]",
                "dh_tmem_L",
            ],
            {
                "transpose_A": "True",
                "clear_accum": "False",
                "mbar": "tcbar_14a",
                "use_2cta": "False",
            },
        ),
        (
            [
                "tmp_shared_2_3",
                "tmp_shared_2_1[:, DK // 2:]",
                "dh_tmem_R",
            ],
            {
                "transpose_A": "True",
                "clear_accum": "False",
                "mbar": "tcbar_14b",
                "use_2cta": "False",
            },
        ),
        (
            [
                "tmp_shared_2_1",
                "tmp_shared_2_3[:, :DV // 2]",
                "dh_tmem_L",
            ],
            {
                "transpose_A": "True",
                "clear_accum": "False",
                "mbar": "tcbar_14a",
                "use_2cta": "False",
            },
        ),
        (
            [
                "tmp_shared_2_1",
                "tmp_shared_2_3[:, DV // 2:]",
                "dh_tmem_R",
            ],
            {
                "transpose_A": "True",
                "clear_accum": "False",
                "mbar": "tcbar_14b",
                "use_2cta": "False",
            },
        ),
    ]

    for args, keywords in signatures:
        assert len(event_lines(events, "tcgen05_gemm", args, keywords)) == 1


def test_root_buffer_name_recognizes_whole_and_sliced_buffers():
    whole = ast.parse("tmp_shared_2_1", mode="eval").body
    sliced = ast.parse("tmp_shared_2_1[:, DK // 2:]", mode="eval").body

    assert _root_buffer_name(whole) == "tmp_shared_2_1"
    assert _root_buffer_name(sliced) == "tmp_shared_2_1"


def test_shared_q_snapshot_is_read_only_through_stage14():
    events = dataflow_events()
    snapshot = event_lines(
        events,
        "copy",
        ["p_fragment", "tmp_shared_2_1[:, DK // 2:]"],
    )[0]
    dot_lines = parallel_multiply_lines(
        "dq_fragment[j_s, j_k]",
        "tmp_shared_2_1[j_s, j_k]",
        ["block_S", "DK"],
    )
    assert len(dot_lines) == 1, "missing full-width shared-Q dot"
    dot = dot_lines[0]
    gemm_12 = event_lines(
        events,
        "tcgen05_gemm",
        ["tmp_shared_1_1", "tmp_shared_2_1", "dk_tmem"],
        {
            "transpose_A": "True",
            "clear_accum": "False",
            "mbar": "tcbar_12",
            "use_2cta": "False",
        },
    )[0]
    stage14_signatures = [
        (
            [
                "tmp_shared_2_3",
                "tmp_shared_2_1[:, :DK // 2]",
                "dh_tmem_L",
            ],
            {
                "transpose_A": "True",
                "clear_accum": "False",
                "mbar": "tcbar_14a",
                "use_2cta": "False",
            },
        ),
        (
            [
                "tmp_shared_2_3",
                "tmp_shared_2_1[:, DK // 2:]",
                "dh_tmem_R",
            ],
            {
                "transpose_A": "True",
                "clear_accum": "False",
                "mbar": "tcbar_14b",
                "use_2cta": "False",
            },
        ),
        (
            [
                "tmp_shared_2_1",
                "tmp_shared_2_3[:, :DV // 2]",
                "dh_tmem_L",
            ],
            {
                "transpose_A": "True",
                "clear_accum": "False",
                "mbar": "tcbar_14a",
                "use_2cta": "False",
            },
        ),
        (
            [
                "tmp_shared_2_1",
                "tmp_shared_2_3[:, DV // 2:]",
                "dh_tmem_R",
            ],
            {
                "transpose_A": "True",
                "clear_accum": "False",
                "mbar": "tcbar_14b",
                "use_2cta": "False",
            },
        ),
    ]
    gemm_14 = [
        event_lines(events, "tcgen05_gemm", args, keywords)[0]
        for args, keywords in stage14_signatures
    ]
    first_gemm_14 = min(gemm_14)
    last_gemm_14 = max(gemm_14)

    wait_09 = max(
        line
        for line in event_lines(
            events, "barrier_wait", ["bar_09", "(i_s + 0) % 2"]
        )
        if snapshot < line < dot
    )
    wait_12 = max(
        line
        for line in event_lines(
            events, "barrier_wait", ["bar_12", "(i_s + 0) % 2"]
        )
        if dot < line < gemm_12
    )
    wait_10 = max(
        line
        for line in event_lines(
            events, "barrier_wait", ["bar_10", "(i_s + 0) % 2"]
        )
        if dot < line < wait_12
    )
    wait_14 = max(
        line
        for line in event_lines(
            events, "barrier_wait", ["bar_14", "(i_s + 0) % 2"]
        )
        if gemm_12 < line < first_gemm_14
    )

    assert (
        snapshot
        < wait_09
        < dot
        < wait_10
        < wait_12
        < gemm_12
        < wait_14
        < first_gemm_14
        <= last_gemm_14
    )
    mutations = buffer_mutations_between(
        "tmp_shared_2_1", snapshot, last_gemm_14
    )
    assert not mutations, "tmp_shared_2_1 is not read-only: {}".format(mutations)


def test_consumer_k_reuses_dk_fragment_and_stages_dead_tmem():
    _, alloc_tmem_targets, alloc_fragment_targets = allocation_targets()
    events = dataflow_events()

    assert len(alloc_tmem_targets) == 10
    assert "odot_fragment_1" not in alloc_fragment_targets
    assert "u_half_fragment" not in alloc_fragment_targets

    u_stage = event_lines(events, "copy", ["u_tmem", "dk_fragment"])
    assert len(u_stage) == 1
    bar_04_wait = max(
        line
        for line in event_lines(
            events, "barrier_wait", ["bar_04", "(i_s + 0) % 2"]
        )
        if line < u_stage[0]
    )
    tcbar_04_wait = next(
        line
        for line in event_lines(
            events, "barrier_wait", ["tcbar_04", "(i_s + 0) % 2"]
        )
        if line > u_stage[0]
    )
    bar_05_arrive = next(
        line
        for line in event_lines(events, "barrier_arrive", ["bar_05"])
        if line > tcbar_04_wait
    )
    assert bar_04_wait < u_stage[0] < tcbar_04_wait < bar_05_arrive

    final_dv = next(
        line
        for line in event_lines(events, "copy", ["dv_tmem", "dv_fragment"])
        if line > bar_05_arrive
    )
    dvg_scale = parallel_multiply_lines(
        "dv_fragment[j_s, j_v]",
        "-g_exp_shared[j_s]",
        ["block_S", "DV"],
    )
    u_dot = parallel_multiply_lines(
        "dk_fragment[j_s, j_v]",
        "dv_fragment[j_s, j_v]",
        ["block_S", "DV"],
    )
    assert len(dvg_scale) == len(u_dot) == 1
    bar_06_arrive = next(
        line
        for line in event_lines(events, "barrier_arrive", ["bar_06"])
        if line > u_dot[0]
    )
    assert final_dv < dvg_scale[0] < u_dot[0] < bar_06_arrive

    bar_06_wait = next(
        line
        for line in event_lines(
            events, "barrier_wait", ["bar_06", "(i_s + 0) % 2"]
        )
        if line > bar_06_arrive
    )
    dvg_stage = event_lines(events, "copy", ["dv_fragment", "u_tmem"])
    u_reduce = event_lines(
        events,
        "reduce_sum",
        ["dk_fragment", "dg_fragment_1"],
        {"dim": "1", "clear": "True"},
    )
    k_stage = event_lines(events, "copy", ["k_shared", "dv_fragment"])
    u_to_dv = event_lines(events, "copy", ["u_tmem", "dv_fragment"])
    assert len(dvg_stage) == len(u_reduce) == len(k_stage) == len(u_to_dv) == 1
    k_tmem_stage = next(
        line
        for line in event_lines(events, "copy", ["dv_fragment", "dv_tmem"])
        if line > k_stage[0]
    )
    bar_08_arrive = next(
        line
        for line in event_lines(events, "barrier_arrive", ["bar_08"])
        if line > u_to_dv[0]
    )
    k_restore = next(
        line
        for line in event_lines(events, "copy", ["dv_tmem", "dv_fragment"])
        if line > bar_08_arrive
    )
    assert (
        bar_06_wait
        < dvg_stage[0]
        < u_reduce[0]
        < k_stage[0]
        < k_tmem_stage
        < u_to_dv[0]
        < bar_08_arrive
        < k_restore
    )

    dk_product = parallel_multiply_lines(
        "dv_fragment[j_s, j_k]",
        "-dk_fragment[j_s, j_k]",
        ["block_S", "DK"],
    )
    assert len(dk_product) == 1
    assert k_restore < dk_product[0]
