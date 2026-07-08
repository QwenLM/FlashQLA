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
    assignments, _, _ = allocation_targets()

    dq_assignment = assignments["dq_tmem"]
    assert isinstance(dq_assignment, ast.Name), ast.dump(dq_assignment)
    assert dq_assignment.id == "u_tmem"


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
        "dq_fragment",
        "db_fragment",
        "dg_fragment_2",
    ]
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


def test_stage09_shared_q_dot_replaces_half_adapters():
    events = dataflow_events()

    dq_store = event_lines(events, "copy", ["dq_fragment", "dq_tmem"])[0]
    dot_lines = parallel_multiply_lines(
        "dq_fragment[j_s, j_k]",
        "tmp_shared_2_1[j_s, j_k]",
        ["block_S", "DK"],
    )
    assert len(dot_lines) == 1, "missing full 64x128 shared-Q dot multiply"
    dot = dot_lines[0]
    dq_reduce = event_lines(
        events,
        "reduce_sum",
        ["dq_fragment", "dg_fragment_2"],
        {"dim": "1", "clear": "False"},
    )[0]
    bar_10 = next(
        line
        for line in event_lines(events, "barrier_arrive", ["bar_10"])
        if line > dq_reduce
    )

    assert not event_lines(
        events,
        "copy",
        ["dq_fragment[:, :DK // 2]", "dp_fragment"],
        required=False,
    )
    assert not event_lines(
        events,
        "copy",
        ["dq_fragment[:, DK // 2:]", "dp_fragment"],
        required=False,
    )
    a_reduces = event_lines(
        events,
        "reduce_sum",
        ["a_fragment", "dg_fragment_2"],
        {"dim": "1", "clear": "False"},
        required=False,
    )
    p_reduces = event_lines(
        events,
        "reduce_sum",
        ["p_fragment", "dg_fragment_2"],
        {"dim": "1", "clear": "False"},
        required=False,
    )
    assert not [line for line in a_reduces if dq_store < line < bar_10]
    assert not [line for line in p_reduces if dq_store < line < bar_10]
    assert dq_store < dot < dq_reduce < bar_10
    assert subscript_accesses_between(
        "tmp_shared_2_1", dq_store, dq_reduce
    ) == [
        {
            "line": dot,
            "expression": _expression("tmp_shared_2_1[j_s, j_k]"),
            "context": "Load",
        }
    ]


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
    dot = parallel_multiply_lines(
        "dq_fragment[j_s, j_k]",
        "tmp_shared_2_1[j_s, j_k]",
        ["block_S", "DK"],
    )[0]
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
    assignments, alloc_tmem_targets, alloc_fragment_targets = allocation_targets()
    events = dataflow_events()

    assert len(alloc_tmem_targets) == 10
    assert "odot_fragment_1" not in alloc_fragment_targets
    assert "u_half_fragment" not in alloc_fragment_targets
    assert isinstance(assignments["dq_tmem"], ast.Name)
    assert assignments["dq_tmem"].id == "u_tmem"

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
