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
_MIN_SUBMIT_ROUND = 4  # 至少探索几回合后才允许交卷
_ACCEPT_DEADLINE = 40  # 白天超过该回合不再接新任务，保证夜晚前回家
_FILE_RE = re.compile(
    r"[A-Za-z_][\w.-]*\.(?:md|txt|json|csv|py|log|yaml|yml|conf|cfg|xml|html|dat|bin)"
)
_REFUSAL_WORDS = ("无法", "不能", "未提供", "不存在", "抱歉", "无法提取", "sorry")
_PATH_RE = re.compile(r"(/[\w./-]{2,})")
_TOKEN_RE = re.compile(r"TOKEN\s*[:=]\s*([A-Za-z0-9_\-]{2,})", re.I)
_JSON_KEY_RE = re.compile(r'\{\s*"([A-Za-z_]\w*)"\s*:')
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
        if ln.startswith(("./", "/", "-", "d", "l")) or "/" in ln.split(" ")[0]
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


def pioneer_action(turn: Turn, state: GameState) -> dict | None:
    pioneer = turn.pioneer()
    if pioneer is None:
        return None
    session = state.task

    if session.active and not turn.phase_task:
        _archive(state)
        session.reset()

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


def attach_exec(turn: Turn, state: GameState) -> str:
    session = state.task
    if not session.active or not turn.phase_task:
        return ""
    if turn.last_cmd_result and session.last_issued:
        session.transcript.append((session.last_issued, turn.last_cmd_result))
        if session.pending_cmds and session.pending_cmds[0] == session.last_issued:
            session.pending_cmds.pop(0)
        session.last_issued = ""
        if len(session.transcript) > _MAX_TRANSCRIPT:
            session.transcript = session.transcript[-_MAX_TRANSCRIPT:]
    if session.text != turn.phase_task and turn.phase_task:
        session.text = turn.phase_task
        _seed_from_sop(state)
    cmd = _next_cmd(turn, state)
    session.last_issued = cmd
    return cmd


def _next_cmd(turn: Turn, state: GameState) -> str:
    session = state.task
    if session.pending_cmds:
        return session.pending_cmds[0]
    done = {cmd for cmd, _result in session.transcript}
    if session.last_issued:
        done.add(session.last_issued)
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
    # 4. 反复跑验收脚本，等 LLM 的修复命令产出真实结果
    check = _check_cmd(session)
    if check is not None:
        return check
    # 5. 通用探索序列
    for cmd in _EXPLORE_CMDS:
        if cmd not in done:
            return cmd
    return _task_driven_cmd(session.text)


def _task_driven_cmd(text: str) -> str:
    paths = re.findall(r"(/[\w./-]+)", text)
    for path in paths:
        if "." in path.rsplit("/", 1)[-1]:
            return f"cat {path}"
    if paths:
        return f"ls -la {paths[0]}"
    return "ls -la && find . -maxdepth 2 -type f | head -20"


def answer_ready(turn: Turn, state: GameState) -> bool:
    session = state.task
    if not session.active or session.step != "EXPLORING":
        return False
    if not session.draft_answer:
        answer = _resolve_answer(session)
        if answer:
            session.draft_answer = answer
    explored = len(session.transcript)
    if session.timeout_rounds:
        remaining = session.accept_round + session.timeout_rounds - turn.round_no
        if remaining <= 3 and session.draft_answer:
            return True
    if _is_selfevo(session.text):
        # 自进化任务必须真跑出结果（如 ./check 的 TOKEN），不拿半成品交卷
        return bool(session.draft_answer)
    if explored < _MIN_SUBMIT_ROUND:
        return False
    # 等 LLM 抽取结果（若还可请求），最多等3回合
    if (
        explored < _MIN_SUBMIT_ROUND + 3
        and not session.llm_extract_requested
        and not session.llm_extract_applied
    ):
        return False
    if session.draft_answer:
        return True
    return explored >= len(_EXPLORE_CMDS) + 6


def submit_action(turn: Turn, state: GameState) -> dict | None:
    session = state.task
    answer = session.draft_answer or _resolve_answer(session)
    if not answer or answer in session.submitted or answer in session.bad_answers:
        return None
    session.submitted.add(answer)
    session.submits += 1
    return submit_command(answer)


def _resolve_answer(session) -> str:
    answer = _token_answer(session)
    if answer:
        return answer
    if _is_selfevo(session.text):
        return ""
    return _text_answer(session)


def _token_answer(session) -> str:
    """从验收脚本输出里取 TOKEN，并按任务要求的 JSON 键名封装。"""
    match = _JSON_KEY_RE.search(session.text)
    key = match.group(1) if match else None
    for _cmd, result in reversed(session.transcript):
        if _echoes_task(result, session.text):
            continue
        for found in _TOKEN_RE.finditer(result):
            value = found.group(1)
            if value.lower() in _PLACEHOLDERS or len(value) < 2:
                continue
            answer = json.dumps({key: value}, ensure_ascii=False) if key else value
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


def handle_answer_error(turn: Turn, state: GameState) -> None:
    session = state.task
    if any(code == 2 for code, _desc in turn.errors):
        if session.draft_answer:
            session.bad_answers.add(session.draft_answer)
        session.draft_answer = ""
        session.llm_extract_requested = False
        session.llm_extract_applied = False


def llm_plan_prompt(state: GameState) -> str:
    session = state.task
    session.llm_plan_requested = True
    session.llm_plan_applied = False
    session.llm_plan_at = len(session.transcript)
    transcript = "\n".join(
        f"$ {cmd}\n{result[:600]}" for cmd, result in session.transcript[-6:]
    )
    return (
        "你在一个无网络的Linux沙盒里用shell命令完成下面的比赛任务。"
        "每条命令都要自带工作目录，例如：cd /path/to/ws && ls -la\n"
        f"任务描述：\n{session.text}\n"
        f"已执行的命令与输出：\n{transcript or '(尚未执行任何命令)'}\n"
        "请给出下一步要执行的shell命令，用于查看文件、修复代码、运行验收脚本，"
        "只输出JSON数组，不要解释：[\"cmd1\",\"cmd2\",...]"
    )


def llm_extract_prompt(state: GameState) -> str:
    session = state.task
    session.llm_extract_requested = True
    transcript = "\n".join(
        f"$ {cmd}\n{result[:500]}" for cmd, result in session.transcript[-8:]
    )
    return (
        "任务描述：\n"
        f"{session.text}\n"
        "沙盒执行记录：\n"
        f"{transcript}\n"
        "请提取任务要求的最终答案，只输出答案本体，不要解释。"
    )


def apply_llm_plan(state: GameState, text: str) -> bool:
    session = state.task
    try:
        import json
        match = re.search(r"\[.*\]", text, re.S)
        if not match:
            return False
        cmds = [str(cmd) for cmd in json.loads(match.group(0))]
        cmds = [cmd for cmd in cmds if cmd and len(cmd) < 500]
        if not cmds:
            return False
        session.pending_cmds = cmds[:8]
        return True
    except Exception:
        return False


def apply_llm_extract(state: GameState, text: str) -> bool:
    session = state.task
    answer = text.strip()
    if len(answer) > 4000:
        return False
    head = answer[:80]
    if any(word in head for word in _REFUSAL_WORDS):
        return False
    if not _plausible(answer, session.text):
        return False
    session.draft_answer = answer
    return True


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
