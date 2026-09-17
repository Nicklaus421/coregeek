import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import logsetup
from .brain import decide

_TRAILING_COMMA = re.compile(r",\s*([}\]])")

logsetup.configure()
LOGGER = logging.getLogger(__name__)
_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = _parse(raw)
            with _LOCK:
                response = decide(payload)
            body = json.dumps(response, ensure_ascii=False).encode("utf-8")
        except Exception:
            # 关键：回包必须合法，异常时给空指令而不是断开连接
            LOGGER.exception("%s decision failed", logsetup.PREFIX)
            body = b'{"roleCommandMap":{},"prompt":"","executeCmd":""}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _parse(raw: bytes) -> dict[str, Any]:
    """先按严格 JSON 解析；失败再容忍尾随逗号这种常见手工格式问题。

    宁可多一次容错也不要因为一个逗号判空回合——空回合算一次异常，
    累计 5 次就被踢出比赛。
    """
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_TRAILING_COMMA.sub(r"\1", text))


def serve(port: int) -> None:
    logsetup.configure()
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
