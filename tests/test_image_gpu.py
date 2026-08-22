from __future__ import annotations

import os
from pathlib import Path

import pytest

from pc_recurrence.image_roi.model import acquire_model, validate_and_load_runtime
from pc_recurrence.runtime import select_device


def test_device_selection_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    device, name = select_device()

    assert device.type == "cpu"
    assert name == "CPU"


@pytest.mark.model
def test_pinned_model_on_available_runtime() -> None:
    if os.environ.get("RUN_IMAGE_MODEL_TEST") != "1":
        pytest.skip("real model smoke test is opt-in")
    model_path = acquire_model(Path(".cache/monai_models"), local_files_only=True)
    _, runtime = validate_and_load_runtime(model_path)
    assert runtime.smoke_test_shape == (4, 3, 96, 96, 96)
