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

from . import logsetup

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
    """进最终日志的是判题器给的 request 原文 + 我方回包；两份也都落 JSONL。

    每回合两行，``req`` 是原样转存的 payload，``resp`` 是最终发出的响应
    （含 roleCommandMap / prompt / executeCmd），便于对着日志复盘任务流程。
    两行都是 json.dumps 的结果，换行会被转义，不会被判题器按行拆散。

    **整体吞异常**：日志绝对不能反过来把决策搞崩（曾因格式化报错导致整回合指令丢失）。
    """
    global _WRITTEN
    try:
        logsetup.emit("req " + json.dumps(payload, ensure_ascii=False))
        logsetup.emit("resp " + json.dumps(response, ensure_ascii=False))
    except Exception:
        pass
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
                    "elapsedMs": round(elapsed_ms, 2),
                    "request": payload,
                    "response": response,
                },
                ensure_ascii=False,
            )
            _FILE.write(line + "\n")
            _WRITTEN += len(line) + 1
    except Exception:
        pass
