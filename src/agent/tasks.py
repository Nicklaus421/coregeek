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
_MAX_TRANSCRIPT = 24
_MIN_SUBMIT_ROUND = 4  # 至少探索几回合后才允许交卷
_ACCEPT_DEADLINE = 40  # 白天超过该回合不再接新任务，保证夜晚前回家
_FILE_RE = re.compile(
    r"[A-Za-z_][\w.-]*\.(?:md|txt|json|csv|py|log|yaml|yml|conf|cfg|xml|html|dat|bin)"
)
_REFUSAL_WORDS = ("无法", "不能", "未提供", "不存在", "抱歉", "无法提取", "sorry")


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
    if len(lines) > 8 and sum(
        1 for ln in lines if ln.startswith(("./", "/proc", "/sys"))
    ) > len(lines) / 2:
        return True
    return False


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
    # 1. 任务文本中提到的文件名 -> 全盘查找
    names = _file_names(session.text)
    for name in names:
        if name not in session.searched_files:
            session.searched_files.add(name)
            return f"find / -name '{name}' 2>/dev/null | head -5"
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
    # 3. 通用探索序列
    if len(session.transcript) < len(_EXPLORE_CMDS):
        return _EXPLORE_CMDS[len(session.transcript)]
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
        answer = _extract_answer(session)
        if answer:
            session.draft_answer = answer
    explored = len(session.transcript)
    if session.timeout_rounds:
        remaining = session.accept_round + session.timeout_rounds - turn.round_no
        if remaining <= 3 and session.draft_answer:
            return True
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
    answer = session.draft_answer or _extract_answer(session) or "N/A"
    if answer in session.submitted:
        return None
    session.submitted.add(answer)
    session.submits += 1
    return submit_command(answer)


def _extract_answer(session) -> str:
    for _cmd, result in reversed(session.transcript):
        lines = [
            line for line in result.splitlines()
            if line.strip() and not line.startswith("[")
        ]
        if not lines:
            continue
        text = "\n".join(lines)[:2000]
        if _is_listing(text):
            continue
        return text
    return ""


def handle_answer_error(turn: Turn, state: GameState) -> None:
    session = state.task
    if any(code == 2 for code, _desc in turn.errors):
        session.draft_answer = ""


def llm_plan_prompt(state: GameState) -> str:
    session = state.task
    session.llm_plan_requested = True
    return (
        "你在一个无网络的Linux沙盒中完成比赛任务。任务描述：\n"
        f"{session.text}\n"
        "请输出完成该任务需要执行的shell/python命令序列，只输出JSON数组："
        '["cmd1","cmd2",...]'
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
    if not answer or len(answer) > 4000:
        return False
    head = answer[:80]
    if any(word in head for word in _REFUSAL_WORDS):
        return False
    session.draft_answer = answer
    return True


def _seed_from_sop(state: GameState) -> None:
    session = state.task
    entry = state.sop_cache.get(task_signature(session.text))
    if entry is not None:
        cmds, answer = entry
        session.pending_cmds = list(cmds)
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
