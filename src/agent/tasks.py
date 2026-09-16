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
    "find . -maxdepth 3 -type f 2>/dev/null | head -50",
    "find . -maxdepth 3 -type f \\( -name '*.md' -o -name '*.txt' -o -name '*.json' -o -name '*.py' \\) 2>/dev/null | head -20 | xargs -I{} sh -c 'echo === {}; head -100 {}'",
)
_MAX_TRANSCRIPT = 24
_MIN_SUBMIT_ROUND = 4  # 至少探索几回合后才允许交卷


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
    if not session.transcript:
        return ""
    result = session.transcript[-1][1]
    lines = [
        line for line in result.splitlines()
        if line.strip() and not line.startswith("[")
    ]
    return "\n".join(lines)[:2000]


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
