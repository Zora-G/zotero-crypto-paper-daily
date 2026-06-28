from __future__ import annotations

"""Minimal HTTP endpoint to collect feedback button clicks from daily emails."""

import argparse
import csv
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


FIELDNAMES = ["timestamp", "action", "source", "title", "paper_url"]
SUCCESS_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Feedback recorded</title>
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2937;
      background: #f8fafc;
    }
    main {
      width: min(92vw, 560px);
      padding: 32px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }
    h1 { margin: 0 0 12px; font-size: 24px; }
    p { margin: 0; line-height: 1.6; color: #4b5563; }
  </style>
</head>
<body>
  <main>
    <h1>反馈已记录</h1>
    <p>谢谢，这条反馈会用于后续论文排序。你可以关闭这个页面。</p>
  </main>
</body>
</html>
"""


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
        def _write_response(self, status: int, body: str, content_type: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            action = qs.get("action", [""])[0]
            title = qs.get("title", [""])[0]
            if action not in {"liked", "dislike"} or not title:
                self._write_response(400, "Missing action/title", "text/plain; charset=utf-8")
                return

            row = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "source": qs.get("source", [""])[0],
                "title": title,
                "paper_url": qs.get("paper_url", [""])[0],
            }
            _append_feedback_row(output, row, retention_days=retention_days)
            self._write_response(200, SUCCESS_HTML, "text/html; charset=utf-8")

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
