from __future__ import annotations

import json
import re
import threading
import webbrowser
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from .review_ui_assets import APP_CSS, APP_JS, INDEX_HTML
from .scan_selection import (
    ScanSelectionConflictError,
    ScanSelectionDocument,
    read_scan_selection,
    update_scan_selections,
)

_MAX_REQUEST_BYTES = 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)
_STATIC_RESPONSES = {
    "/": ("text/html; charset=utf-8", INDEX_HTML.encode()),
    "/assets/app.css": ("text/css; charset=utf-8", APP_CSS.encode()),
    "/assets/app.js": ("text/javascript; charset=utf-8", APP_JS.encode()),
}


class _ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    selection_path: Path
    write_lock: threading.Lock
    review_url: str


class _ReviewRequestHandler(BaseHTTPRequestHandler):
    server: _ReviewServer

    def _request_path(self) -> str | None:
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            return None
        return parsed.path

    def _is_known_path(self, path: str | None) -> bool:
        return bool(
            path in {*_STATIC_RESPONSES, "/api/inventory", "/api/selections"}
            or (path and path.startswith("/api/previews/"))
        )

    def _request_is_local(self) -> bool:
        expected_host = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != expected_host:
            self._send_error(HTTPStatus.FORBIDDEN, "unexpected Host header")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{expected_host}":
            self._send_error(HTTPStatus.FORBIDDEN, "foreign Origin is not allowed")
            return False
        return True

    def _send_headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Security-Policy", _CONTENT_SECURITY_POLICY)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()

    def _send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self._send_headers(status, content_type, len(body))
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self._send_bytes(status, "application/json; charset=utf-8", body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message})

    def _method_not_allowed(self) -> None:
        path = self._request_path()
        if not self._request_is_local():
            return
        if self._is_known_path(path):
            self._send_error(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_GET(self) -> None:  # noqa: N802
        if not self._request_is_local():
            return
        path = self._request_path()
        if path in _STATIC_RESPONSES:
            content_type, body = _STATIC_RESPONSES[path]
            self._send_bytes(HTTPStatus.OK, content_type, body)
            return
        if path == "/api/inventory":
            self._serve_inventory()
            return
        if path and path.startswith("/api/previews/"):
            self._serve_preview(path.removeprefix("/api/previews/"))
            return
        if path == "/api/selections":
            self._send_error(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._request_is_local():
            return
        path = self._request_path()
        if path == "/api/selections":
            self._update_selections()
            return
        if self._is_known_path(path):
            self._send_error(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_HEAD(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def _serve_inventory(self) -> None:
        try:
            document = read_scan_selection(self.server.selection_path)
            payload = _inventory_payload(document)
        except ValueError as exc:
            self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            return
        except OSError as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"unable to read inventory: {exc}")
            return
        self._send_json(HTTPStatus.OK, payload)

    def _serve_preview(self, identifier: str) -> None:
        if _SHA256_PATTERN.fullmatch(identifier) is None:
            self._send_error(HTTPStatus.NOT_FOUND, "preview not found")
            return
        try:
            document = read_scan_selection(self.server.selection_path)
        except ValueError as exc:
            self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            return
        except OSError as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"unable to read inventory: {exc}")
            return
        row = next(
            (item for item in document.rows if item["candidate_id"] == identifier),
            None,
        )
        if row is None:
            self._send_error(HTTPStatus.NOT_FOUND, "preview not found")
            return
        preview_path, failure = _resolved_preview(self.server.selection_path, row)
        if preview_path is None:
            self._send_error(HTTPStatus.NOT_FOUND, failure)
            return
        try:
            body = preview_path.read_bytes()
        except OSError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, f"preview unavailable: {exc}")
            return
        self._send_bytes(HTTPStatus.OK, "image/png", body)

    def _read_json_payload(self) -> dict[str, Any] | None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Content-Type must be application/json",
            )
            return None
        raw_length = self.headers.get("Content-Length")
        try:
            content_length = int(raw_length or "")
        except ValueError:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            return None
        if content_length < 0:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            return None
        if content_length > _MAX_REQUEST_BYTES:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body exceeds 1 MiB")
            return None
        body = self.rfile.read(content_length)
        if len(body) != content_length:
            self._send_error(HTTPStatus.BAD_REQUEST, "incomplete request body")
            return None
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error(HTTPStatus.BAD_REQUEST, "malformed JSON")
            return None
        if not isinstance(payload, dict):
            self._send_error(HTTPStatus.BAD_REQUEST, "JSON payload must be an object")
            return None
        return payload

    def _update_selections(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        if set(payload) != {"expected_sha256", "selections"}:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "payload must contain exactly expected_sha256 and selections",
            )
            return
        expected_sha256 = payload["expected_sha256"]
        selections = payload["selections"]
        valid_revision = (
            isinstance(expected_sha256, str)
            and _SHA256_PATTERN.fullmatch(expected_sha256) is not None
        )
        if not valid_revision:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "expected_sha256 must be 64 lowercase hex characters",
            )
            return
        if not isinstance(selections, dict) or not all(
            isinstance(patient_id, str)
            and (identifier is None or isinstance(identifier, str))
            and (identifier is None or _SHA256_PATTERN.fullmatch(identifier) is not None)
            for patient_id, identifier in selections.items()
        ):
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "selections must map patient IDs to candidate IDs or null",
            )
            return
        try:
            with self.server.write_lock:
                revision = update_scan_selections(
                    self.server.selection_path,
                    cast(dict[str, str | None], selections),
                    expected_sha256=expected_sha256,
                )
        except ScanSelectionConflictError as exc:
            self._send_error(HTTPStatus.CONFLICT, str(exc))
            return
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except OSError as exc:
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"unable to update scan selection: {exc}",
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "selection_sha256": revision,
                "selected_patient_count": sum(value is not None for value in selections.values()),
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        return


def _resolved_preview(
    selection_path: Path,
    row: dict[str, str],
) -> tuple[Path | None, str]:
    recorded_reason = row["preview_reason"].strip()
    if row["preview_status"] != "ready":
        return None, recorded_reason or "preview was not generated"
    relative_path = row["preview_path"].strip()
    if not relative_path:
        return None, recorded_reason or "preview path is missing"
    review_directory = selection_path.parent.resolve()
    preview_path = (selection_path.parent / relative_path).resolve()
    try:
        preview_path.relative_to(review_directory)
    except ValueError:
        return None, recorded_reason or "preview path is outside the review directory"
    if not preview_path.is_file():
        return None, recorded_reason or "preview file is missing"
    return preview_path, ""


def _blocked_reason(rows: list[dict[str, str]]) -> str:
    sentinel = next((row for row in rows if row["status"] == "no_series"), None)
    if sentinel is not None:
        reason = sentinel["reason"].strip() or "No CT Series found"
        image_range = sentinel["image_range"].strip()
        return f"{reason} Requested range: {image_range}." if image_range else reason
    reasons = list(dict.fromkeys(row["reason"].strip() for row in rows if row["reason"].strip()))
    return "; ".join(reasons) or "No ready CT Series is available"


def _inventory_payload(document: ScanSelectionDocument) -> dict[str, Any]:
    patients: list[dict[str, Any]] = []
    issue_patient_count = 0
    for patient_id in document.patient_ids:
        rows = [row for row in document.rows if row["patient_id"].strip() == patient_id]
        candidate_rows = [row for row in rows if row["status"] != "no_series"]
        ready_count = sum(row["status"] == "ready" for row in candidate_rows)
        selected_candidate_id = document.selected_candidate_ids[patient_id]
        state = "selected" if selected_candidate_id else "unselected" if ready_count else "blocked"
        candidates: list[dict[str, Any]] = []
        has_issue = state == "blocked"
        for row in candidate_rows:
            candidate = {key: value for key, value in row.items() if key != "selected"}
            candidate["is_selected"] = row["candidate_id"] == selected_candidate_id
            preview_path, _ = _resolved_preview(document.selection_path, row)
            candidate["preview_url"] = (
                f"/api/previews/{row['candidate_id']}" if preview_path is not None else ""
            )
            candidates.append(candidate)
            has_issue = has_issue or bool(
                row["status"] == "not_selectable"
                or row["geometry_warnings"].strip()
                or preview_path is None
            )
        if has_issue:
            issue_patient_count += 1
        patients.append(
            {
                "patient_id": patient_id,
                "dicom_folder": rows[0]["dicom_folder"],
                "state": state,
                "selected_candidate_id": selected_candidate_id,
                "ready_candidate_count": ready_count,
                "candidate_count": len(candidate_rows),
                "blocked_reason": _blocked_reason(rows) if state == "blocked" else "",
                "candidates": candidates,
            }
        )

    selectable_count = sum(patient["ready_candidate_count"] > 0 for patient in patients)
    selected_count = sum(patient["selected_candidate_id"] is not None for patient in patients)
    blocked_count = sum(patient["state"] == "blocked" for patient in patients)
    return {
        "selection_sha256": document.selection_sha256,
        "selection_file": str(document.selection_path.resolve()),
        "counts": {
            "patient_count": len(patients),
            "selectable_patient_count": selectable_count,
            "selected_patient_count": selected_count,
            "blocked_patient_count": blocked_count,
            "issue_patient_count": issue_patient_count,
        },
        "patients": patients,
    }


def create_review_server(
    selection_path: Path,
    *,
    port: int = 8765,
) -> ThreadingHTTPServer:
    read_scan_selection(selection_path)
    server = _ReviewServer(("127.0.0.1", port), _ReviewRequestHandler)
    server.selection_path = selection_path
    server.write_lock = threading.Lock()
    server.review_url = f"http://127.0.0.1:{server.server_port}"
    return server


def serve_scan_review(
    selection_path: Path,
    *,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    server = create_review_server(selection_path, port=port)
    timer: threading.Timer | None = None
    try:
        print(f"CT Series review: {server.review_url}")
        print("Press Ctrl+C to stop.")
        if open_browser:
            timer = threading.Timer(0.2, webbrowser.open, args=(server.review_url,))
            timer.daemon = True
            timer.start()
        with suppress(KeyboardInterrupt):
            server.serve_forever()
    finally:
        if timer is not None:
            timer.cancel()
        server.server_close()
