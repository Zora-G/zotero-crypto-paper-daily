from __future__ import annotations

"""Minimal HTTP endpoint to collect feedback button clicks from daily emails."""

import argparse
import csv
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


FIELDNAMES = ["timestamp", "action", "source", "title", "paper_url"]


def _parse_timestamp(raw_timestamp: str) -> datetime | None:
    if not raw_timestamp:
        return None
    try:
        return datetime.fromisoformat(raw_timestamp)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(raw_timestamp, fmt)
            return parsed
        except ValueError:
            continue
    return None


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fp:
        yield from csv.DictReader(fp)


def _append_feedback_row(path: Path, row: dict[str, str], retention_days: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if path.exists():
        try:
            for row_data in _iter_rows(path):
                if set(row_data.keys()) != set(FIELDNAMES):
                    continue
                rows.append(row_data)
        except Exception:
            rows = []

    rows.append(row)

    if retention_days and retention_days > 0:
        cutoff = datetime.now() - timedelta(days=retention_days)
        rows = [
            r
            for r in rows
            if (ts := _parse_timestamp(r.get("timestamp", ""))) is None or ts >= cutoff
        ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def serve_feedback_server(host: str, port: int, output: Path, retention_days: int | None) -> None:
    class FeedbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            action = qs.get("action", [""])[0]
            title = qs.get("title", [""])[0]
            if action not in {"liked", "dislike"} or not title:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing action/title")
                return

            row = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "source": qs.get("source", [""])[0],
                "title": title,
                "paper_url": qs.get("paper_url", [""])[0],
            }
            _append_feedback_row(output, row, retention_days=retention_days)
            self.send_response(204)
            self.end_headers()

        def log_message(self, _format: str, *args) -> None:  # pragma: no cover
            pass

    server = HTTPServer((host, port), FeedbackHandler)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--out", default="feedback.csv", help="Path to CSV feedback history file.")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="Keep only records generated in the last N days (0 to keep forever).",
    )
    args = parser.parse_args()
    retention = args.retention_days
    if retention is not None and retention <= 0:
        retention = None

    serve_feedback_server(args.host, args.port, Path(args.out).expanduser(), retention)


if __name__ == "__main__":
    main()
