"""自进化类任务的执行闭环。

任务是一个"判题器中转的三角关系"：本进程 <-> 判题器 <-> 大模型/沙盒。
两条通道都是**一回合延迟**的：

    本回合 response.executeCmd  ->  下回合 request.lastCmdResult
    本回合 response.prompt      ->  下回合 request.llmResp

所以这里的核心是一个跨回合的状态机（``TaskSession``），而不是单回合函数：
每回合把回来的结果记进 transcript，再决定下一条命令 / 下一个提问。

两条硬约束（踩过的坑）：
- 判题器只在"发了命令"时才回 ``lastCmdResult``，且**命令输出为空是常态**
  （``sed -i``/重定向等成功命令都没有输出）。所以推进只看"上一回合确实发过
  命令"，绝不能拿结果内容判断，否则同一条命令会被无限重发、整局卡死。
- 离开任务点一格 / 开拓者死亡 / 超时都会让任务直接作废，所以任务期间
  开拓者必须钉死在任务点上（夜间同理，见 ``combat.night_commands``）。
"""
import json
import re

from .grid import step_adjacent
from .protocol import (
    Turn,
    accept_command,
    distance,
    move_command,
    submit_command,
)
from .state import GameState

LLM_PLAN = "plan"
LLM_EXTRACT = "extract"
_LLM_TTL = 3  # 提问超过这么多回合还没回就当丢了，别把队列堵死

_EXPLORE_CMDS = (
    "pwd && ls -la",
    "ls -la /tmp /home /root 2>/dev/null",
    "find /tmp /home /root /opt /srv /data /workspace . -maxdepth 4 -type f 2>/dev/null | head -60",
)
_DISCOVER_CMDS = (
    "ls -la /tmp 2>/dev/null | head -30",
    "find / -maxdepth 4 -name 'spec.md' 2>/dev/null | head -10",
    "find / -maxdepth 5 -type d -name 'ws_*' 2>/dev/null | head -10",
)
_MAX_TRANSCRIPT = 24
_MAX_RESULT = 4000  # 沙盒输出可达 64KB，落 transcript 前先截断
_CHECK_EVERY = 4  # 命令用尽后每隔几回合重跑一次验收脚本
_ACCEPT_DEADLINE = 40  # 白天超过该回合不再接新任务，保证夜晚前回家
_FILE_RE = re.compile(
    r"[A-Za-z_][\w.-]*\.(?:md|txt|json|csv|py|log|yaml|yml|conf|cfg|xml|html|dat|bin)"
)
_REFUSAL_WORDS = ("无法", "不能", "未提供", "不存在", "抱歉", "无法提取", "sorry")
_PATH_RE = re.compile(r"(/[\w./-]{2,})")
_TOKEN_RE = re.compile(r"TOKEN\s*[:=]\s*([A-Za-z0-9_\-]{2,})", re.I)
_JSON_KEY_RE = re.compile(r'\{\s*"([A-Za-z_]\w*)"\s*:')
_SHELL_HINT_RE = re.compile(
    r"(^|\s)(cd|ls|cat|find|grep|sed|awk|head|tail|echo|chmod|chown|cp|mv|rm|"
    r"python3?|pip3?|bash|sh|env|export|make|gcc|node|\./)"
)
_PLACEHOLDER_RE = re.compile(
    r"\b(x{2,}|unknown|todo|tbd|n/?a|none|null|your[_-]?token|placeholder)\b",
    re.I,
)
_PLACEHOLDERS = frozenset(
    {"xx", "xxx", "unknown", "todo", "tbd", "n/a", "na", "none", "null", "无"}
)
_ERROR_MARKERS = (
    "no such file or directory", "command not found", "permission denied",
    "cannot access", "is a directory", "traceback (most recent call last)",
    "syntaxerror", "modulenotfounderror", "importerror",
    "notimplementederror", "assertionerror", "/bin/sh:", "/bin/bash:",
    "[timeout]", "[judger_error]",
)
_ERROR_HEADS = ("error", "failed", "fail:", "cannot", "unable", "traceback")
_SELFEVO_HINTS = ("./check", "selfevolution", "ws_")
_TOKEN_WORDS = ("token", "令牌", "凭证")
_ACTION_HINTS = ("运行", "执行", "修复", "脚本", "沙盒", "run ", "./")
_LISTING_HEADS = ("./", "/", "-", "d", "l", "total")


def _file_names(text: str) -> list[str]:
    names: list[str] = []
    for name in _FILE_RE.findall(text):
        short = name.rsplit("/", 1)[-1]
        if short not in names:
            names.append(short)
    return names


def _is_listing(result: str) -> bool:
    if "drwx" in result or "total " in result[:200]:
        return True
    lines = [ln for ln in result.splitlines() if ln.strip()]
    if len(lines) < 5:
        return False
    pathish = sum(
        1 for ln in lines
        if ln.startswith(_LISTING_HEADS) or "/" in ln.split(" ")[0]
    )
    return pathish > len(lines) / 2


def _is_selfevo(text: str) -> bool:
    """自进化类任务：要在沙盒里跑脚本 / 修代码，答案是一次性的（token 等）。"""
    low = text.lower()
    if any(hint in low for hint in _SELFEVO_HINTS):
        return True
    has_token = any(word in low for word in _TOKEN_WORDS)
    has_action = any(hint in low for hint in _ACTION_HINTS)
    return has_token and has_action


def _workspace(text: str) -> str | None:
    paths = [path.rstrip("/") for path in _PATH_RE.findall(text)]
    for path in paths:
        if re.search(r"/ws_\d+$", path) or "selfevolution" in path.lower():
            return path
    for path in paths:
        if path.count("/") >= 3 and "." not in path.rsplit("/", 1)[-1]:
            return path
    return None


def _workspace_from_transcript(session) -> str | None:
    for _cmd, result in reversed(session.transcript):
        match = re.search(r"(/[\w./-]*?/ws_\d+)", result)
        if match:
            return match.group(1)
        match = re.search(r"(/[\w./-]+)/spec\.md", result)
        if match:
            return match.group(1)
    return None


def _check_cmd(session) -> str | None:
    ws = _workspace(session.text) or _workspace_from_transcript(session)
    if ws is None:
        return None
    return f"cd {ws} && (./check 2>&1 || true) | tail -30"


def _chore(session) -> list[str]:
    """自进化类任务的固定 SOP：定位工作区 -> 读说明 -> 看验收脚本报错。"""
    ws = _workspace(session.text) or _workspace_from_transcript(session)
    if ws is None:
        return list(_DISCOVER_CMDS)
    return [
        f"ls -la {ws}",
        f"cat {ws}/spec.md 2>/dev/null | head -120",
        f"cd {ws} && find . -maxdepth 2 -type f -not -path './.git/*' | head -40",
        f"cd {ws} && (./check 2>&1 || true) | tail -30",
    ]


def _looks_like_error(text: str) -> bool:
    low = text.lower()
    if any(marker in low for marker in _ERROR_MARKERS):
        return True
    head = low[:120]
    return any(word in head for word in _ERROR_HEADS)


def _echoes_task(text: str, task_text: str) -> bool:
    """任务原文（spec.md 的内容）不是答案。"""
    body = re.sub(r"\s+", "", text)
    target = re.sub(r"\s+", "", task_text)
    if len(body) < 8 or not target:
        return False
    return body[:120] in target


def _placeholder_only(text: str) -> bool:
    scrubbed = re.sub(r'"[^"]*"\s*:', "", text)
    scrubbed = re.sub(r"[A-Za-z_]\w*\s*[:=]", "", scrubbed)
    scrubbed = _PLACEHOLDER_RE.sub("", scrubbed)
    scrubbed = re.sub(r"[\s{}\[\]\"',:]+", "", scrubbed)
    return len(scrubbed) < 3


def _plausible(text: str, task_text: str) -> bool:
    body = text.strip()
    if not body or len(body) > 2000:
        return False
    if _looks_like_error(body) or _is_listing(body):
        return False
    if _echoes_task(body, task_text) or _placeholder_only(body):
        return False
    stripped = body.strip("{}[]\"' \n\t").strip()
    return len(stripped) >= 2 and stripped.lower() not in _PLACEHOLDERS


def task_signature(text: str) -> str:
    head = re.sub(r"\s+", "", text)[:40]
    return head


# ---------------------------------------------------------------- 任务生命周期


def handle_turn(turn: Turn, state: GameState) -> None:
    """任务是否已经结束（phaseTask 清空 / 开拓者阵亡）——与昼夜无关。

    刚领取的一两个回合内 phaseTask 可能还没下发，给一个宽限期，否则会把
    刚建立的任务会话立刻清掉。
    """
    session = state.task
    if not session.active:
        return
    pioneer_alive = turn.pioneer() is not None
    if pioneer_alive and (
        turn.phase_task or turn.round_no - session.accept_round < 2
    ):
        return
    _archive(state)
    session.reset()


def pioneer_action(turn: Turn, state: GameState) -> dict | None:
    pioneer = turn.pioneer()
    if pioneer is None:
        return None
    session = state.task
    if session.active:
        session.step = "EXPLORING"
        return None  # 原地不动；executeCmd 由 attach_exec 输出
    point = _pick_point(turn)
    if point is None:
        return None
    if distance(pioneer.pos, point.pos) <= 1:
        session.active = True
        session.point_pos = point.pos
        session.accept_round = turn.round_no
        session.timeout_rounds = point.timeout_rounds
        session.step = "ACCEPTED"
        return accept_command()
    session.step = "GOTO"
    session.point_pos = point.pos
    step = step_adjacent(turn, pioneer, point.pos)
    if step is None:
        return None
    return move_command(step)


def _pick_point(turn: Turn):
    if turn.day_round > _ACCEPT_DEADLINE:
        return None
    candidates = [
        point for point in turn.task_points
        if point.is_valid and point.cooldown == 0
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: -(p.score_reward / max(1, p.timeout_rounds)),
    )
    return candidates[0]


# ------------------------------------------------------------ executeCmd 通道


def attach_exec(turn: Turn, state: GameState) -> str:
    """推进沙盒命令队列：先收上一回合的结果，再发下一条。"""
    session = state.task
    if not session.active or not turn.phase_task:
        return ""
    if session.text != turn.phase_task:
        session.text = turn.phase_task
        _seed_from_sop(state)
    if session.last_issued and turn.round_no > session.last_issued_round:
        # 只要上一回合确实发过命令，就认为结果已经回来（哪怕是空输出）
        _record(session, session.last_issued, turn.last_cmd_result)
        session.last_issued = ""
    cmd = _next_cmd(turn, state)
    session.last_issued = cmd
    session.last_issued_round = turn.round_no
    return cmd


def _record(session, cmd: str, result: str) -> None:
    session.transcript.append((cmd, (result or "")[:_MAX_RESULT]))
    if session.pending_cmds and session.pending_cmds[0] == cmd:
        session.pending_cmds.pop(0)
    if len(session.transcript) > _MAX_TRANSCRIPT:
        session.transcript = session.transcript[-_MAX_TRANSCRIPT:]


def _next_cmd(turn: Turn, state: GameState) -> str:
    session = state.task
    if session.pending_cmds:
        return session.pending_cmds[0]
    done = {cmd for cmd, _result in session.transcript}
    # 1. 任务文本中提到的文件名 -> 全盘查找
    names = _file_names(session.text)
    for name in names:
        if name in session.searched_files:
            continue
        session.searched_files.add(name)
        cmd = f"find / -name '{name}' 2>/dev/null | head -5"
        if cmd not in done:
            return cmd
    # 2. 上一次 find 找到了路径 -> 读取内容
    if session.transcript:
        last_cmd, last_res = session.transcript[-1]
        if last_cmd.startswith("find /"):
            for line in last_res.splitlines():
                line = line.strip()
                if line.startswith("/") and any(
                    line.endswith(name) for name in names
                ):
                    return f"cat {line}"
    # 3. 模式化任务的固定剧本（自进化类）
    for cmd in _chore(session):
        if cmd not in done:
            return cmd
    # 4. 验收脚本：换过命令就立刻重跑；否则每 _CHECK_EVERY 回合探一次，
    #    这样命令用尽后还有兜底通道（沙盒状态可能被外部改变 / LLM 迟到）
    check = _check_cmd(session)
    if check is not None:
        last = session.transcript[-1][0] if session.transcript else ""
        if (
            check not in done
            or last != check
            or turn.round_no - session.check_round >= _CHECK_EVERY
        ):
            session.check_round = turn.round_no
            return check
    # 5. 通用探索序列
    for cmd in _EXPLORE_CMDS:
        if cmd not in done:
            return cmd
    fallback = _task_driven_cmd(session.text)
    return fallback if fallback not in done else ""


def _task_driven_cmd(text: str) -> str:
    paths = re.findall(r"(/[\w./-]+)", text)
    for path in paths:
        if "." in path.rsplit("/", 1)[-1]:
            return f"cat {path}"
    if paths:
        return f"ls -la {paths[0]}"
    return "ls -la && find . -maxdepth 2 -type f | head -20"


# --------------------------------------------------------------- LLM 通道


def task_prompt(turn: Turn, state: GameState) -> str:
    """任务期间向大模型提问：先要命令序列，拿到执行结果后再让它抽答案。"""
    session = state.task
    if not session.active:
        return ""
    _purge_inflight(session, turn.round_no)
    if session.pending_cmds or session.llm_inflight:
        return ""
    if turn.round_no - session.llm_asked_round < 2:
        return ""
    if session.draft_answer:
        return ""
    session.llm_asked_round = turn.round_no
    if not session.llm_plan_done:
        session.llm_inflight.append((LLM_PLAN, turn.round_no))
        return llm_plan_prompt(state)
    session.llm_inflight.append((LLM_EXTRACT, turn.round_no))
    return llm_extract_prompt(state)


def apply_llm_resp(turn: Turn, state: GameState, text: str) -> bool:
    """把回来的 llmResp 折进任务状态。

    不严格按提问类型分发：LLM 有时会跳过"给命令"直接给答案，也有时把答案
    包装成命令。先看能不能当命令用，再退回去当答案，两头都不浪费。
    """
    session = state.task
    if not session.active:
        return False
    _purge_inflight(session, turn.round_no)
    kind = session.llm_inflight.pop(0)[0] if session.llm_inflight else LLM_EXTRACT
    text = text.strip()
    if not text:
        return False
    cmds = _as_commands(text)
    if cmds is not None and kind == LLM_PLAN:
        session.pending_cmds = cmds
        session.llm_plan_done = True
        return True
    if _apply_answer(state, text):
        return True
    if cmds is not None:
        session.pending_cmds = cmds
        session.llm_plan_done = True
        return True
    return False


def _purge_inflight(session, round_no: int) -> None:
    session.llm_inflight = [
        entry for entry in session.llm_inflight
        if round_no - entry[1] <= _LLM_TTL
    ]


def _as_commands(text: str) -> list[str] | None:
    """只有"看着像 shell 命令的 JSON 字符串数组"才当命令，避免吃掉答案。"""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None
    if not all(isinstance(item, str) for item in items):
        return None
    if any(not item or len(item) > 500 for item in items):
        return None
    if not any(_SHELL_HINT_RE.search(item) for item in items):
        return None
    return items[:8]


def _apply_answer(state: GameState, text: str) -> bool:
    session = state.task
    if len(text) > 4000:
        return False
    head = text[:80].lower()
    if any(word in head for word in _REFUSAL_WORDS):
        return False
    if text in session.bad_answers or text in session.submitted:
        return False
    if not _plausible(text, session.text):
        return False
    session.draft_answer = text
    return True


def llm_plan_prompt(state: GameState) -> str:
    session = state.task
    transcript = "\n".join(
        f"$ {cmd}\n{(result or '(无输出)')[:600]}"
        for cmd, result in session.transcript[-6:]
    )
    return (
        "你在一个无网络的 Linux 沙盒里用 shell 命令完成下面的比赛任务。"
        "每条命令都要自带工作目录，例如：cd /path/to/ws && ls -la\n"
        f"任务描述：\n{session.text}\n"
        f"已执行的命令与输出：\n{transcript or '(尚未执行任何命令)'}\n"
        "请给出下一步要执行的 shell 命令，用于查看文件、修复代码、运行验收脚本，"
        "只输出 JSON 数组，不要解释：[\"cmd1\",\"cmd2\",...]"
    )


def llm_extract_prompt(state: GameState) -> str:
    session = state.task
    transcript = "\n".join(
        f"$ {cmd}\n{(result or '(无输出)')[:500]}"
        for cmd, result in session.transcript[-8:]
    )
    rejected = ""
    if session.bad_answers:
        rejected = "以下答案已被判题器判定为错误，不要再给：\n" + "\n".join(
            sorted(session.bad_answers)[:5]
        ) + "\n"
    return (
        "任务描述：\n"
        f"{session.text}\n"
        "沙盒执行记录：\n"
        f"{transcript}\n"
        f"{rejected}"
        "请提取任务要求的最终答案，只输出答案本体（若任务要求 JSON 就输出 JSON），"
        "不要解释、不要复述任务原文、不要输出命令。"
    )


def _resolve_answer(session) -> str:
    answer = _token_answer(session)
    if answer:
        return answer
    if _is_selfevo(session.text):
        return ""  # 自进化任务必须真跑出结果，不能拿半成品交卷
    return _text_answer(session)


def _answer_re(session) -> re.Pattern | None:
    """任务要求 JSON 时，按它声明的键名去 transcript 里找对应取值。"""
    match = _JSON_KEY_RE.search(session.text)
    if match:
        key = re.escape(match.group(1))
        return re.compile(
            rf"{key}\s*[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-]{{2,}})", re.I,
        )
    return None


def _token_answer(session) -> str:
    match = _JSON_KEY_RE.search(session.text)
    key = match.group(1) if match else None
    patterns = [_TOKEN_RE]
    declared = _answer_re(session)
    if declared is not None:
        patterns.append(declared)
    for _cmd, result in reversed(session.transcript):
        if _echoes_task(result, session.text):
            continue
        for pattern in patterns:
            for found in pattern.finditer(result):
                value = found.group(1)
                if value.lower() in _PLACEHOLDERS or len(value) < 2:
                    continue
                answer = (
                    json.dumps({key: value}, ensure_ascii=False) if key else value
                )
                if answer not in session.bad_answers:
                    return answer
    return ""


def _text_answer(session) -> str:
    for _cmd, result in reversed(session.transcript):
        lines = [
            line for line in result.splitlines()
            if line.strip() and not line.startswith("[")
        ]
        if not lines:
            continue
        body = "\n".join(lines)[:2000]
        if _plausible(body, session.text):
            return body
    return ""


# ---------------------------------------------------------------- 交卷


def answer_ready(turn: Turn, state: GameState) -> bool:
    """有可信答案就交——判题器按"历史最高通过率"记分，早交不会更差。"""
    session = state.task
    if not session.active or session.step != "EXPLORING":
        return False
    answer = session.draft_answer or _resolve_answer(session)
    if not answer:
        return False
    session.draft_answer = answer
    return answer not in session.submitted and answer not in session.bad_answers


def submit_action(turn: Turn, state: GameState) -> dict | None:
    session = state.task
    answer = session.draft_answer
    if not answer or answer in session.submitted or answer in session.bad_answers:
        return None
    session.submitted.add(answer)
    session.submits += 1
    return submit_command(answer)


def handle_answer_error(turn: Turn, state: GameState) -> None:
    """errorCode=2：答案错了。任务不会因此结束，记下来继续换答案。"""
    session = state.task
    if not any(code == 2 for code, _desc in turn.errors):
        return
    if session.draft_answer:
        session.bad_answers.add(session.draft_answer)
    session.draft_answer = ""
    _purge_inflight(session, turn.round_no)


def _seed_from_sop(state: GameState) -> None:
    session = state.task
    entry = state.sop_cache.get(task_signature(session.text))
    if entry is None:
        return
    cmds, answer = entry
    session.pending_cmds = list(cmds)
    # 自进化类任务的答案每次都不一样，只复用命令流程
    if not _is_selfevo(session.text):
        session.draft_answer = answer


def _archive(state: GameState) -> None:
    session = state.task
    if session.text:
        cmds = [cmd for cmd, _r in session.transcript[:6]]
        state.sop_cache[task_signature(session.text)] = (
            cmds, session.draft_answer,
        )
    if len(state.sop_cache) > 32:
        state.sop_cache.pop(next(iter(state.sop_cache)))
