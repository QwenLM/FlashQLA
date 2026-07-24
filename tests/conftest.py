import pytest
import torch

requires_gpu = pytest.mark.gpu
requires_hopper = pytest.mark.hopper
requires_blackwell = pytest.mark.blackwell
requires_sm120 = pytest.mark.sm120

GPU_AVAILABLE = torch.cuda.is_available()

ARCH = None
if GPU_AVAILABLE:
    try:
        import tilelang.contrib.nvcc
        _cv = tilelang.contrib.nvcc.get_target_compute_version()
        if _cv == "9.0":
            ARCH = "SM90"
        elif _cv == "10.0":
            ARCH = "SM100"
        elif _cv == "10.3":
            ARCH = "SM103"
        elif _cv == "12.0":
            ARCH = "SM120"
    except Exception:
        pass


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not GPU_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="CUDA GPU not available"))
        if "hopper" in item.keywords and ARCH != "SM90":
            item.add_marker(pytest.mark.skip(reason="Hopper (SM90) GPU required"))
        if "blackwell" in item.keywords and ARCH not in ["SM100", "SM103"]:
            item.add_marker(pytest.mark.skip(reason="Blackwell (SM100 or SM103) GPU required"))
        if "sm120" in item.keywords and ARCH != "SM120":
            item.add_marker(pytest.mark.skip(reason="Blackwell SM120 GPU required"))
