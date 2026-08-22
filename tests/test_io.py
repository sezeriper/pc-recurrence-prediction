from __future__ import annotations

import hashlib
from pathlib import Path

from pc_recurrence.io import create_run_directory, sha256_file, write_summary


def test_sha256_file_matches_stdlib_lowercase_digest(tmp_path: Path) -> None:
    path = tmp_path / 'payload.bin'
    payload = b'abc123' * 100_000
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_write_summary_serializes_only_requested_columns(tmp_path: Path) -> None:
    summary = tmp_path / 'summary.csv'
    rows = [
        {
            'patient_id': 'Patient 1',
            'status': 'detected',
            'shape': (8, 9, 10),
            'count': 3,
            'extra': 'ignored',
        }
    ]

    write_summary(rows, summary, columns=('patient_id', 'status', 'shape', 'count'))

    assert summary.read_text(encoding='utf-8-sig').splitlines() == [
        'patient_id,status,shape,count',
        'Patient 1,detected,8x9x10,3',
    ]
    assert not summary.with_suffix(summary.suffix + '.tmp').exists()


def test_create_run_directory_generates_unique_names(tmp_path: Path) -> None:
    first = create_run_directory(tmp_path)
    second = create_run_directory(tmp_path)

    assert first != second
    assert first.is_dir() and second.is_dir()
