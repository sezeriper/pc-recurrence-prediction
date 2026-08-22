from __future__ import annotations

from pc_recurrence.io import create_run_directory, write_json, write_npz, write_summary

__all__ = ["create_run_directory", "write_json", "write_npz", "write_summary"]

EMBEDDING_SUMMARY_COLUMNS = (
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
)
