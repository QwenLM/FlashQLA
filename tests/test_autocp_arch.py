# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tilelang.contrib.nvcc
import torch

from flash_qla.ops.gated_delta_rule.chunk.cp.autocp.decision import decide
from flash_qla.ops.gated_delta_rule.chunk.cp.autocp import launch, utils
from flash_qla.ops.gated_delta_rule.chunk.cp.autocp.launch import current_arch
from flash_qla.ops.gated_delta_rule.chunk.cp.autocp.utils import (
    AutocpInfo,
    is_calibrated,
    load_coefs,
)


@pytest.mark.parametrize(
    "version, arch",
    [
        ("9.0", "sm90"),
        ("10.0", "sm100"),
        ("10.3", "sm103"),
        ("12.0", "sm120"),
        ("12.1", "sm121"),
    ],
)
def test_autocp_recognizes_supported_architectures(monkeypatch, version, arch):
    monkeypatch.setattr(
        tilelang.contrib.nvcc, "get_target_compute_version", lambda: version
    )
    current_arch.cache_clear()
    try:
        assert current_arch() == arch
    finally:
        current_arch.cache_clear()


def test_context_initializes_on_sm121(monkeypatch):
    monkeypatch.setattr(
        tilelang.contrib.nvcc, "get_target_compute_version", lambda: "12.1"
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args: SimpleNamespace(multi_processor_count=48),
    )
    name = "flash_qla.ops.gated_delta_rule.chunk.cp._test_sm121_context"
    spec = importlib.util.spec_from_file_location(
        name, Path(launch.__file__).parents[1] / "context.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    current_arch.cache_clear()
    try:
        spec.loader.exec_module(module)
        assert module.ARCH == "sm121"
        assert module.AUTOCP_MODEL
    finally:
        current_arch.cache_clear()


@pytest.mark.parametrize("is_train", [False, True])
def test_sm121_autocp_uses_sm120_coefficients(is_train, tmp_path, monkeypatch):
    shutil.copyfile(Path(utils.COEFS_DIR) / "sm120.csv", tmp_path / "sm120.csv")
    monkeypatch.setattr(utils, "COEFS_DIR", str(tmp_path))
    assert is_calibrated("sm121")
    with pytest.warns(AutocpInfo, match="no sm121.csv; using sm120's coefficients"):
        coefs = load_coefs(arch="sm121")
    assert coefs["backend"] == "blackwell_sm120"
    assert coefs["chunk"] == 32
    use_cp, local_chunks = decide(
        num_chunks=[256], num_v_heads=8, P=48, coefs=coefs, is_train=is_train
    )
    assert isinstance(use_cp, bool)
    assert local_chunks is None or local_chunks > 0


def test_reference_helpers_use_kernel_chunk_size():
    from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE
    from gdr_common import CHUNK_SIZE as reference_chunk_size

    assert reference_chunk_size == CHUNK_SIZE
