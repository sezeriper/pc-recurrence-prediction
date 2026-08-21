from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def create_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    candidate = output_root / base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def write_npz(path: Path, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return path


def write_summary(rows: list[dict[str, Any]], path: Path) -> Path:
    columns = [
        "patient_id",
        "status",
        "reason",
        "encoder",
        "roi_target",
        "native_roi_shape",
        "resampled_shape",
        "patch_count",
        "embedding_dimension",
        "inference_seconds",
        "roi_ct_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            for key in ("native_roi_shape", "resampled_shape"):
                value = serialized.get(key)
                if value is not None and not isinstance(value, str):
                    serialized[key] = "x".join(str(item) for item in value)
            writer.writerow(serialized)
    return path
