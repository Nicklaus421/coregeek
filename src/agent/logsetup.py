"""统一日志出口：把决策过程打到 stdout，供判题器收进最终日志。

判题器通过 HTTP 调用本服务，stdout/stderr 不承载协议，可以安全占用；
服务进程由判题器拉起，它的 stdout/stderr 一般会被重定向进最终日志文件，
所以"要出现在最终日志里的内容"就往这里写。

三条约定：
- 每回合只输出**一行**紧凑摘要（前缀 [agent] 写进消息体里，便于 grep，
  这样即使判题器/入口自己配了 formatter，前缀也不会丢）；
- 强制行缓冲，且 logging 的 StreamHandler 每条都会 flush，
  避免进程被 kill 时块缓冲里的内容整段丢失；
- 只用 logging，不直接 print，避免两套格式混在一起。

完整 request/response 仍落到 logs/agent_trace.jsonl（见 trace.py）。

开关：``COREGEEK_LOG=0`` 关闭 stdout 输出（默认开启）。
"""
from __future__ import annotations

import logging
import os
import sys

PREFIX = "[agent]"
_ENABLED = os.environ.get("COREGEEK_LOG", "1") != "0"
_LOGGER = logging.getLogger("agent.round")


def configure() -> None:
    """幂等配置：行缓冲 + 默认 INFO 级输出到 stdout。

    任意入口都可以重复调用（main3.py 已经 basicConfig 过一次也没关系），
    只在 root 没有任何 handler 时才挂 handler，避免日志重复。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass
    if not _ENABLED:
        return
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        root.addHandler(handler)


def emit(text: str) -> None:
    """打一行到最终日志。text 内部不要再带换行。写日志失败绝不影响决策。"""
    if not _ENABLED:
        return
    try:
        _LOGGER.info("%s %s", PREFIX, text)
    except Exception:
        pass


def banner(text: str) -> None:
    """启动横幅：确认"我的输出确实进了这份日志"，并报出 JSONL 路径。"""
    emit(text)
