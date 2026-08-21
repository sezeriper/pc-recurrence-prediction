from __future__ import annotations

import os
from pathlib import Path

import pytest

from pc_recurrence.image_roi.model import acquire_model, validate_and_load_runtime


@pytest.mark.gpu
def test_pinned_model_on_verified_rocm_runtime() -> None:
    if os.environ.get("RUN_IMAGE_GPU_TEST") != "1":
        pytest.skip("real ROCm model smoke test is opt-in")
    model_path = acquire_model(Path(".cache/monai_models"), local_files_only=True)
    _, runtime = validate_and_load_runtime(model_path)
    assert runtime.smoke_test_shape == (4, 3, 96, 96, 96)
