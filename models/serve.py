"""Стенд: шаг считает models.law, страница только рисует."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from models.law import Stand

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "universe-pressure-3d.html"
STAND = Stand()
HOST = "127.0.0.1"
PORT = 8766


def pace() -> None:
    while True:
        if STAND.paused:
            time.sleep(0.05)
            continue
        started = time.perf_counter()
        advanced = 0.0
        for _ in range(40):
            if STAND.paused:
                break
            advanced += STAND.step()
            if advanced >= 0.02 or time.perf_counter() - started > 0.03:
                break
        rest = 0.03 - (time.perf_counter() - started)
        if rest > 0.0:
            time.sleep(rest)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, code: int, body: bytes, content: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/api/state"):
            body = json.dumps(STAND.snapshot()).encode("utf-8")
            self._send(200, body, "application/json")
            return
        if self.path in ("/", "/universe-pressure-3d.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/control":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if "paused" in payload:
            STAND.paused = bool(payload["paused"])
        if payload.get("reset") or "n" in payload or "pressure" in payload:
            STAND.reset(payload.get("n"), payload.get("pressure"))
        body = json.dumps(STAND.snapshot()).encode("utf-8")
        self._send(200, body, "application/json")


def main() -> None:
    threading.Thread(target=pace, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"http://{HOST}:{PORT}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
