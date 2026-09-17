"""把每回合的请求/响应落盘成 JSONL，供赛后复盘与调参。

设计约束：
- 绝不能拖慢决策（5 秒上限）：只做一次 json.dumps + 一次 write，不 fsync；
- 绝不影响主流程：任何异常都吞掉；
- 文件按大小滚动，保留一份历史，避免长局把磁盘写满。

开关（环境变量）：
- ``COREGEEK_TRACE=0``   关闭（默认开启）
- ``COREGEEK_TRACE_DIR`` 指定输出目录（默认 <仓库根>/logs）
"""
from __future__ import annotations

import json
import os
import threading
import time

_ENABLED = os.environ.get("COREGEEK_TRACE", "1") != "0"
_DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)
_DIR = os.environ.get("COREGEEK_TRACE_DIR") or _DEFAULT_DIR
_NAME = "agent_trace.jsonl"
_MAX_BYTES = 16 * 1024 * 1024

_LOCK = threading.Lock()
_FILE = None
_WRITTEN = 0


def _open():
    global _FILE, _WRITTEN
    os.makedirs(_DIR, exist_ok=True)
    path = os.path.join(_DIR, _NAME)
    _WRITTEN = os.path.getsize(path) if os.path.exists(path) else 0
    _FILE = open(path, "a", encoding="utf-8", buffering=1)
    return path, _WRITTEN


def _rotate():
    global _FILE
    path = os.path.join(_DIR, _NAME)
    _FILE.close()
    backup = path + ".1"
    if os.path.exists(backup):
        os.remove(backup)
    os.rename(path, backup)
    _FILE = open(path, "a", encoding="utf-8", buffering=1)


def record(
    payload: dict,
    response: dict,
    elapsed_ms: float,
    request_id: str = "",
) -> None:
    """写一条回合记录。payload/response 为原始报文，附一条便于 grep 的摘要。"""
    global _WRITTEN
    if not _ENABLED:
        return
    try:
        with _LOCK:
            if _FILE is None:
                _open()
            elif _WRITTEN >= _MAX_BYTES:
                _rotate()
                _WRITTEN = 0
            line = json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "requestId": request_id,
                    "roundNo": payload.get("roundNo"),
                    "elapsedMs": round(elapsed_ms, 2),
                    "summary": _summary(payload, response),
                    "request": payload,
                    "response": response,
                },
                ensure_ascii=False,
            )
            _FILE.write(line + "\n")
            _WRITTEN += len(line) + 1
    except Exception:
        pass


def _summary(payload: dict, response: dict) -> dict:
    team = payload.get("teamOur") or {}
    robots = ((payload.get("robot") or {}).get("roles")) or []
    commands = response.get("roleCommandMap") or {}
    return {
        "gold": team.get("goldNum"),
        "score": team.get("totalScore"),
        "side": team.get("type"),
        "roles": [
            {
                "id": role.get("id"),
                "type": role.get("roleType"),
                "x": (role.get("pos") or {}).get("x"),
                "y": (role.get("pos") or {}).get("y"),
                "hp": role.get("health"),
                "level": role.get("level"),
                "bag": role.get("backpack"),
            }
            for role in team.get("roles") or ()
        ],
        "robotCount": len(robots),
        "actions": {uid: cmd.get("action") for uid, cmd in commands.items()},
        "errors": [
            err.get("errorCode") for err in payload.get("errors") or ()
        ],
        "hasPrompt": bool(response.get("prompt")),
        "executeCmd": response.get("executeCmd"),
    }
