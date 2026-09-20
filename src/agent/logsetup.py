"""日志与留痕。

约定：
- stdout 每回合只打印一行原始 payload（便于在判题器日志里查看判题器到底给了什么）。
- 完整 request / response 落盘到 ``logs/agent_trace.jsonl``，每行一个事件。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# 仓库根目录 = 本文件上三级（src/agent/logsetup.py -> 仓库根）。
_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _ROOT / "logs"
_TRACE_PATH = _LOG_DIR / "agent_trace.jsonl"


def setup() -> None:
    """初始化 stdout 日志（main3.py 已做，这里保证幂等）。"""
    _LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
    )


def emit(payload: dict) -> None:
    """把判题器下发的原始 payload 打印到 stdout（唯一 stdout 日志行）。"""
    logging.info("[agent] %s", json.dumps(payload, ensure_ascii=False))


def trace(kind: str, data: dict) -> None:
    """完整留痕到 jsonl。"""
    with open(_TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, **data}, ensure_ascii=False, default=str) + "\n")
