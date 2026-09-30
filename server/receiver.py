"""AI Graver report receiver: accepts zip reports from installed copies and stores them on disk.

Only uploads are possible: there is no way to list or download reports over HTTP.
Config (environment): AIGRAVER_UPLOAD_TOKEN, AIGRAVER_DATA_DIR, AIGRAVER_PORT.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOKEN = os.environ["AIGRAVER_UPLOAD_TOKEN"]
DATA_DIR = Path(os.environ.get("AIGRAVER_DATA_DIR", "/var/lib/aigraver-reports"))
PORT = int(os.environ.get("AIGRAVER_PORT", "3200"))
MAX_BYTES = 60 * 1024 * 1024
INSTALL_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class Handler(BaseHTTPRequestHandler):
    server_version = "aigraver-reports"

    def do_POST(self) -> None:
        if self.path != "/upload":
            return self._reply(404, "not found")
        if not hmac.compare_digest(self.headers.get("X-Upload-Token", ""), TOKEN):
            return self._reply(403, "forbidden")
        install_id = self.headers.get("X-Install-Id", "")
        if not INSTALL_ID.match(install_id):
            return self._reply(400, "bad install id")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BYTES:
            return self._reply(413, "bad size")
        body = self.rfile.read(length)
        if not body.startswith(b"PK"):
            return self._reply(400, "zip expected")
        folder = DATA_DIR / install_id
        folder.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d_%H%M%S") + ".zip"
        (folder / name).write_bytes(body)
        meta = {
            "version": self.headers.get("X-App-Version", "")[:20],
            "computer": self.headers.get("X-Computer", "")[:80],
            "last_upload": name,
        }
        (folder / "install.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return self._reply(200, "ok")

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._reply(200, "ok")
        return self._reply(404, "not found")

    def _reply(self, code: int, text: str) -> None:
        payload = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.headers.get('X-Real-IP', self.client_address[0])} {fmt % args}", flush=True)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
