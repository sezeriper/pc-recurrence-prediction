from __future__ import annotations

import pytest

from pc_recurrence.runtime import select_device


def test_device_selection_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    device, name = select_device()

    assert device.type == "cpu"
    assert name == "CPU"
