from __future__ import annotations

import csv
import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pc_recurrence.image_roi import cli as image_cli
from pc_recurrence.image_roi import review_ui
from pc_recurrence.image_roi.review_ui import create_review_server
from pc_recurrence.image_roi.review_ui_assets import APP_CSS, APP_JS, INDEX_HTML
from pc_recurrence.image_roi.scan_selection import (
    SCAN_SELECTION_COLUMNS,
    ScanSelectionConflictError,
    read_scan_selection,
    update_scan_selections,
)


def _candidate(
    patient_id: str,
    identifier: str,
    *,
    status: str = "ready",
    selected: str = "",
    preview_status: str = "ready",
    preview_path: str = "",
    reason: str = "",
) -> dict[str, str]:
    row = {column: "" for column in SCAN_SELECTION_COLUMNS}
    row.update(
        {
            "selected": selected,
            "patient_id": patient_id,
            "dicom_folder": patient_id.replace(" ", "").upper(),
            "candidate_id": identifier,
            "status": status,
            "reason": reason,
            "study_instance_uid": f"1.2.840.{identifier[0]}",
            "series_instance_uid": f"1.2.840.{identifier[0]}.1",
            "sop_class_uid": "1.2.840.10008.5.1.4.1.1.2",
            "source_directories": f"{patient_id}/source",
            "source_file_count": "12",
            "unique_file_count": "10",
            "duplicate_file_count": "2",
            "series_sop_uids_sha256": identifier,
            "study_date": "20240102",
            "study_description": "Pancreas study",
            "series_number": "7",
            "acquisition_number": "2",
            "series_description": f"Portal venous {identifier[0]}",
            "protocol_name": "Abdomen contrast",
            "body_part_examined": "ABDOMEN",
            "contrast_bolus_agent": "Iohexol",
            "rows": "512",
            "columns": "512",
            "slice_thickness_mm": "2.5",
            "row_spacing_mm": "0.8",
            "column_spacing_mm": "0.8",
            "image_range": "1-9",
            "selected_file_count": "9",
            "median_slice_spacing_mm": "2.5",
            "maximum_slice_gap_mm": "3.0",
            "geometry_warnings": "maximum slice gap exceeds median",
            "preview_status": preview_status,
            "preview_reason": "" if preview_status == "ready" else "render failed",
            "preview_path": preview_path,
        }
    )
    return row


def _review_rows() -> list[dict[str, str]]:
    no_series = {column: "" for column in SCAN_SELECTION_COLUMNS}
    no_series.update(
        {
            "patient_id": "Patient 3",
            "dicom_folder": "PATIENT3",
            "status": "no_series",
            "reason": "No CT Series found in the requested range",
            "image_range": "4-12",
        }
    )
    return [
        _candidate("Patient 1", "a" * 64, preview_path="previews/a.png"),
        _candidate("Patient 1", "b" * 64, preview_path="previews/b.png"),
        _candidate(
            "Patient 1",
            "c" * 64,
            status="not_selectable",
            reason="unsupported SOP Class UID",
            preview_path="previews/c.png",
        ),
        _candidate(
            "Patient 2",
            "d" * 64,
            preview_path="previews/missing.png",
        ),
        no_series,
    ]


def _write_selection(path: Path, rows: list[dict[str, str]], *, columns: Any = None) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or SCAN_SELECTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_read_and_update_scan_selection_preserve_audit_cells(tmp_path: Path) -> None:
    selection = tmp_path / "scan_selection.csv"
    original_rows = _review_rows()
    _write_selection(selection, original_rows)

    document = read_scan_selection(selection)
    assert document.patient_ids == ("Patient 1", "Patient 2", "Patient 3")
    assert document.selected_candidate_ids == {
        "Patient 1": None,
        "Patient 2": None,
        "Patient 3": None,
    }

    revision = update_scan_selections(
        selection,
        {"Patient 1": "a" * 64, "Patient 2": None, "Patient 3": None},
        expected_sha256=document.selection_sha256,
    )
    assert selection.read_bytes().startswith(b"\xef\xbb\xbf")
    saved_rows = _read_rows(selection)
    assert [row["selected"] for row in saved_rows] == ["yes", "", "", "", ""]
    for original, saved in zip(original_rows, saved_rows, strict=True):
        assert {key: value for key, value in saved.items() if key != "selected"} == {
            key: value for key, value in original.items() if key != "selected"
        }

    revision = update_scan_selections(
        selection,
        {"Patient 1": "b" * 64, "Patient 2": None, "Patient 3": None},
        expected_sha256=revision,
    )
    assert [row["selected"] for row in _read_rows(selection)] == ["", "yes", "", "", ""]

    update_scan_selections(
        selection,
        {"Patient 1": None, "Patient 2": None, "Patient 3": None},
        expected_sha256=revision,
    )
    assert all(not row["selected"] for row in _read_rows(selection))


@pytest.mark.parametrize("case", ["multiple", "non_ready", "duplicate", "bad_candidate"])
def test_read_scan_selection_rejects_malformed_current_choices(tmp_path: Path, case: str) -> None:
    selection = tmp_path / "scan_selection.csv"
    rows = _review_rows()
    if case == "multiple":
        rows[0]["selected"] = "yes"
        rows[1]["selected"] = " YES "
    elif case == "non_ready":
        rows[2]["selected"] = "yes"
    elif case == "duplicate":
        rows[1]["candidate_id"] = rows[0]["candidate_id"]
    else:
        rows[0]["candidate_id"] = "NOT-A-SHA"
    _write_selection(selection, rows)
    before = selection.read_bytes()

    with pytest.raises(ValueError, match="Patient 1|patient Patient 1"):
        read_scan_selection(selection)
    assert selection.read_bytes() == before


def test_update_scan_selection_rejects_invalid_choices_without_mutation(tmp_path: Path) -> None:
    selection = tmp_path / "scan_selection.csv"
    _write_selection(selection, _review_rows())
    document = read_scan_selection(selection)
    before = selection.read_bytes()

    invalid_selections = [
        {"Patient 1": "e" * 64, "Patient 2": None, "Patient 3": None},
        {"Patient 1": "c" * 64, "Patient 2": None, "Patient 3": None},
        {"Patient 1": None, "Patient 2": None},
    ]
    for choices in invalid_selections:
        with pytest.raises(ValueError):
            update_scan_selections(
                selection,
                choices,
                expected_sha256=document.selection_sha256,
            )
        assert selection.read_bytes() == before


def test_scan_selection_schema_and_stale_revision_fail_without_mutation(tmp_path: Path) -> None:
    selection = tmp_path / "scan_selection.csv"
    rows = _review_rows()
    malformed_columns = SCAN_SELECTION_COLUMNS[:-1]
    _write_selection(
        selection,
        [{key: value for key, value in row.items() if key in malformed_columns} for row in rows],
        columns=malformed_columns,
    )
    malformed = selection.read_bytes()
    with pytest.raises(ValueError, match="Unexpected scan selection schema"):
        read_scan_selection(selection)
    assert selection.read_bytes() == malformed

    _write_selection(selection, rows)
    before = selection.read_bytes()
    with pytest.raises(
        ScanSelectionConflictError,
        match="^scan selection changed on disk; reload before saving$",
    ):
        update_scan_selections(
            selection,
            {"Patient 1": None, "Patient 2": None, "Patient 3": None},
            expected_sha256="0" * 64,
        )
    assert selection.read_bytes() == before


@pytest.fixture
def running_review_server(
    tmp_path: Path,
) -> Iterator[tuple[Path, Any]]:
    selection = tmp_path / "scan_selection.csv"
    _write_selection(selection, _review_rows())
    previews = tmp_path / "previews"
    previews.mkdir()
    for name in ("a", "b", "c"):
        (previews / f"{name}.png").write_bytes(b"\x89PNG\r\nsynthetic")
    server = create_review_server(selection, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield selection, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _http_request(
    server: Any,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    response_body = response.read()
    response_headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, response_headers, response_body


def _post_json(
    server: Any,
    payload: Any,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload).encode()
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    return _http_request(
        server,
        "POST",
        "/api/selections",
        body=body,
        headers=request_headers,
    )


def test_review_inventory_and_assets_preserve_order_and_candidate_states(
    running_review_server: tuple[Path, Any],
) -> None:
    selection, server = running_review_server
    for path, expected_type in (
        ("/", "text/html"),
        ("/assets/app.css", "text/css"),
        ("/assets/app.js", "text/javascript"),
    ):
        status, headers, body = _http_request(server, "GET", path)
        assert status == 200
        assert headers["content-type"].startswith(expected_type)
        assert body
        assert headers["content-security-policy"].startswith("default-src 'self'")
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"
        assert "access-control-allow-origin" not in headers

    status, _, body = _http_request(server, "GET", "/api/inventory")
    assert status == 200
    payload = json.loads(body)
    assert payload["selection_file"] == str(selection.resolve())
    assert payload["counts"] == {
        "patient_count": 3,
        "selectable_patient_count": 2,
        "selected_patient_count": 0,
        "blocked_patient_count": 1,
        "issue_patient_count": 3,
    }
    assert [patient["patient_id"] for patient in payload["patients"]] == [
        "Patient 1",
        "Patient 2",
        "Patient 3",
    ]
    assert [patient["candidate_count"] for patient in payload["patients"]] == [3, 1, 0]
    assert [candidate["status"] for candidate in payload["patients"][0]["candidates"]] == [
        "ready",
        "ready",
        "not_selectable",
    ]
    missing_preview_patient = payload["patients"][1]
    assert missing_preview_patient["state"] == "unselected"
    assert missing_preview_patient["ready_candidate_count"] == 1
    assert missing_preview_patient["candidates"][0]["preview_status"] == "ready"
    assert missing_preview_patient["candidates"][0]["preview_url"] == ""
    blocked = payload["patients"][2]
    assert blocked["state"] == "blocked"
    assert blocked["candidates"] == []
    assert "4-12" in blocked["blocked_reason"]


def test_review_preview_routes_are_confined_and_preserve_png_bytes(
    running_review_server: tuple[Path, Any],
) -> None:
    _, server = running_review_server
    for identifier in ("a" * 64, "b" * 64, "c" * 64):
        status, headers, body = _http_request(
            server,
            "GET",
            f"/api/previews/{identifier}",
        )
        assert status == 200
        assert headers["content-type"] == "image/png"
        assert body == b"\x89PNG\r\nsynthetic"

    for path in (
        f"/api/previews/{'d' * 64}",
        f"/api/previews/{'e' * 64}",
        "/api/previews/../../scan_selection.csv",
        "/api/previews/%2e%2e%2fscan_selection.csv",
    ):
        status, _, body = _http_request(server, "GET", path)
        assert status == 404
        assert "error" in json.loads(body)


def test_review_selection_post_updates_partial_progress_and_rejects_stale_write(
    running_review_server: tuple[Path, Any],
) -> None:
    selection, server = running_review_server
    status, _, body = _http_request(server, "GET", "/api/inventory")
    revision = json.loads(body)["selection_sha256"]
    choices = {
        "Patient 1": "a" * 64,
        "Patient 2": "d" * 64,
        "Patient 3": None,
    }
    status, _, body = _post_json(
        server,
        {"expected_sha256": revision, "selections": choices},
    )
    assert status == 200
    result = json.loads(body)
    assert result["selected_patient_count"] == 2
    assert result["selection_sha256"] != revision
    assert [row["selected"] for row in _read_rows(selection)] == [
        "yes",
        "",
        "",
        "yes",
        "",
    ]

    saved = selection.read_bytes()
    status, _, body = _post_json(
        server,
        {
            "expected_sha256": revision,
            "selections": {**choices, "Patient 1": "b" * 64},
        },
    )
    assert status == 409
    assert json.loads(body) == {"error": "scan selection changed on disk; reload before saving"}
    assert selection.read_bytes() == saved


def test_review_http_validation_rejects_bad_requests_without_mutation(
    running_review_server: tuple[Path, Any],
) -> None:
    selection, server = running_review_server
    document = read_scan_selection(selection)
    valid_choices = {"Patient 1": None, "Patient 2": None, "Patient 3": None}
    before = selection.read_bytes()

    invalid_requests = [
        _http_request(
            server,
            "POST",
            "/api/selections",
            body=b"{",
            headers={"Content-Type": "application/json"},
        ),
        _post_json(server, {"expected_sha256": document.selection_sha256}),
        _post_json(
            server,
            {
                "expected_sha256": document.selection_sha256,
                "selections": {**valid_choices, "Patient 1": "c" * 64},
            },
        ),
        _http_request(
            server,
            "POST",
            "/api/selections",
            body=b"{}",
            headers={"Content-Type": "text/plain"},
        ),
        _post_json(
            server,
            {
                "expected_sha256": document.selection_sha256,
                "selections": valid_choices,
            },
            headers={"Origin": "https://foreign.example"},
        ),
        _post_json(
            server,
            {
                "expected_sha256": document.selection_sha256,
                "selections": valid_choices,
            },
            headers={"Host": "localhost:8765"},
        ),
    ]
    assert [status for status, _, _ in invalid_requests] == [400, 400, 400, 415, 403, 403]
    assert selection.read_bytes() == before

    status, _, _ = _http_request(
        server,
        "POST",
        "/api/selections",
        body=b"",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(1024 * 1024 + 1),
        },
    )
    assert status == 413
    assert selection.read_bytes() == before
    assert _http_request(server, "PUT", "/api/inventory")[0] == 405
    assert _http_request(server, "GET", "/unknown")[0] == 404


def test_external_csv_corruption_blocks_inventory_and_saving(
    running_review_server: tuple[Path, Any],
) -> None:
    selection, server = running_review_server
    revision = read_scan_selection(selection).selection_sha256
    selection.write_text("not,the,inventory\n", encoding="utf-8")
    corrupted = selection.read_bytes()

    status, _, _ = _http_request(server, "GET", "/api/inventory")
    assert status == 422
    status, _, _ = _post_json(
        server,
        {
            "expected_sha256": revision,
            "selections": {"Patient 1": None, "Patient 2": None, "Patient 3": None},
        },
    )
    assert status == 400
    assert selection.read_bytes() == corrupted


def test_fixed_review_assets_expose_accessible_workflow_controls() -> None:
    assert "Save selections" in INDEX_HTML
    assert "Next unselected" in INDEX_HTML
    assert 'aria-modal="true"' in INDEX_HTML
    assert "100%" in INDEX_HTML
    assert "minmax(min(100%, 360px), 1fr)" in APP_CSS
    assert "beforeunload" in APP_JS
    assert "textContent" in APP_JS
    assert "innerHTML" not in APP_JS
    assert "!candidate.preview_url" in APP_JS


def test_review_cli_help_and_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = tmp_path / "scan_selection.csv"
    _write_selection(selection, _review_rows())
    captured: dict[str, Any] = {}

    def fake_serve(
        selection_path: Path,
        *,
        port: int,
        open_browser: bool,
    ) -> None:
        captured.update(
            selection_path=selection_path,
            port=port,
            open_browser=open_browser,
        )

    monkeypatch.setattr(review_ui, "serve_scan_review", fake_serve)
    runner = CliRunner()

    help_result = runner.invoke(image_cli.app, ["review", "--help"])
    assert help_result.exit_code == 0
    assert "--selection" in help_result.output
    assert "--port" in help_result.output
    assert "--open-browser" in help_result.output
    assert "--no-open-browser" in help_result.output
    assert "Review montages locally" in help_result.output
    assert "montage reviewer" in help_result.output
    assert "selected" in help_result.output
    assert "values" in help_result.output

    result = runner.invoke(
        image_cli.app,
        [
            "review",
            "--selection",
            str(selection),
            "--port",
            "9123",
            "--no-open-browser",
        ],
    )
    assert result.exit_code == 0
    assert captured == {
        "selection_path": selection.resolve(),
        "port": 9123,
        "open_browser": False,
    }


def test_review_cli_startup_error_exits_one_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = tmp_path / "scan_selection.csv"
    _write_selection(selection, _review_rows())
    before = selection.read_bytes()

    def fail_startup(
        _selection_path: Path,
        *,
        port: int,
        open_browser: bool,
    ) -> None:
        raise OSError(f"cannot bind port {port} (open_browser={open_browser})")

    monkeypatch.setattr(review_ui, "serve_scan_review", fail_startup)
    result = CliRunner().invoke(
        image_cli.app,
        ["review", "--selection", str(selection), "--no-open-browser"],
    )
    assert result.exit_code == 1
    assert "cannot bind port 8765" in result.output
    assert selection.read_bytes() == before
