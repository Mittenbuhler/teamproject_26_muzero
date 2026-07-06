"""Serve the dashboard locally and persist editable state-MCTS run names."""

from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse
import webbrowser

from .dashboard import generate_dashboard


def write_report_atomic(report_path, report):
    report_path = Path(report_path)
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_path.replace(report_path)


def rename_run(report_path, run_id, display_name):
    """Atomically update one run's optional display name while preserving run_id."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Run renaming requires a schema-version-2 report with a runs list")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(display_name, str):
        raise ValueError("display_name must be a string")
    display_name = display_name.strip()
    if len(display_name) > 100:
        raise ValueError("display_name cannot exceed 100 characters")

    matches = [run for run in runs if run.get("run_id") == run_id]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one run with id {run_id!r}, found {len(matches)}")
    run = matches[0]
    if display_name:
        run["display_name"] = display_name
    else:
        run.pop("display_name", None)

    write_report_atomic(report_path, report)
    return run


def update_folders(report_path, folders, run_folders):
    """Persist ordered dashboard folders and each run's optional assignment."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Folders require a schema-version-2 report with a runs list")
    if not isinstance(folders, list) or not isinstance(run_folders, dict):
        raise ValueError("folders must be a list and run_folders must be an object")

    normalized_folders = []
    folder_ids = set()
    for folder in folders:
        if not isinstance(folder, dict):
            raise ValueError("Each folder must be an object")
        folder_id = folder.get("id")
        name = folder.get("name")
        if not isinstance(folder_id, str) or not folder_id.strip():
            raise ValueError("Each folder id must be a non-empty string")
        folder_id = folder_id.strip()
        if folder_id in folder_ids:
            raise ValueError("Folder ids cannot contain duplicates")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each folder name must be a non-empty string")
        name = name.strip()
        if len(folder_id) > 100 or len(name) > 100:
            raise ValueError("Folder ids and names cannot exceed 100 characters")
        folder_ids.add(folder_id)
        normalized_folders.append({"id": folder_id, "name": name})

    run_ids = {run.get("run_id") for run in runs}
    normalized_assignments = {}
    for run_id, folder_id in run_folders.items():
        if not isinstance(run_id, str) or not isinstance(folder_id, str):
            raise ValueError("Run-folder assignments must map string ids to string ids")
        if run_id not in run_ids:
            raise KeyError(f"Unknown run id: {run_id!r}")
        if folder_id not in folder_ids:
            raise KeyError(f"Unknown folder id: {folder_id!r}")
        normalized_assignments[run_id] = folder_id

    report["dashboard_folders"] = normalized_folders
    report["dashboard_run_folders"] = normalized_assignments
    write_report_atomic(report_path, report)
    return normalized_folders, normalized_assignments


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, artifact_dir, **kwargs):
        self.artifact_dir = Path(artifact_dir)
        super().__init__(*args, directory=str(self.artifact_dir), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if urlparse(self.path).path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/diagnostics_dashboard.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        endpoint = urlparse(self.path).path
        if endpoint not in {"/api/run-name", "/api/folders"}:
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 65536:
                raise ValueError("Request body must contain at most 65536 bytes")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            report_path = self.artifact_dir / "state_mcts_report.json"
            if endpoint == "/api/run-name":
                run = rename_run(
                    report_path,
                    payload.get("run_id"),
                    payload.get("display_name"),
                )
                response_payload = {
                    "run_id": run["run_id"],
                    "display_name": run.get("display_name", ""),
                }
            else:
                folders, assignments = update_folders(
                    report_path,
                    payload.get("folders"),
                    payload.get("run_folders"),
                )
                response_payload = {
                    "dashboard_folders": folders,
                    "dashboard_run_folders": assignments,
                }
            generate_dashboard(
                self.artifact_dir,
                self.artifact_dir / "diagnostics_dashboard.html",
            )
            self._json_response(HTTPStatus.OK, response_payload)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except OSError as error:
            self._json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def _json_response(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/state_mcts")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the editable dashboard in the default browser after startup.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    generate_dashboard(artifact_dir, artifact_dir / "diagnostics_dashboard.html")
    handler = partial(DashboardHandler, artifact_dir=artifact_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"editable dashboard: {url}")
    print("press Ctrl+C to stop")
    if args.open_browser:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
