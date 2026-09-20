"""HTTP 服务：接收判题器请求，调用 Agent 决策，返回指令 JSON。"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import logsetup
from .brain import Agent

_AGENT = Agent()


class _Handler(BaseHTTPRequestHandler):
    server_version = "CoreGeekAgent/0.1"

    def do_POST(self) -> None:  # noqa: N802 (http.server 约定)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._reply(400, {"error": "invalid json"})
            return

        logsetup.emit(payload)
        logsetup.trace("request", {"payload": payload})

        try:
            response = _AGENT.decide(payload)
        except Exception as exc:  # noqa: BLE001 兜底，避免进程崩溃
            logsetup.trace("error", {"error": str(exc)})
            response = {"roleCommandMap": {}, "prompt": "", "executeCmd": ""}

        logsetup.trace("response", {"payload": payload, "response": response})
        self._reply(200, response)

    def do_GET(self) -> None:  # noqa: N802
        self._reply(200, {"status": "ok"})

    def _reply(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # 关闭默认访问日志，避免污染判题器日志
        return


def serve(port: int) -> None:
    logsetup.setup()
    httpd = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    httpd.serve_forever()
