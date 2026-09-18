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
    r"\b(x{2,}|unknown|todo|tbd|n/?a|none|null|your[_-]?token|placeholder)\b"
    r"|<[^<>]{0,40}>",
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


_EXIT_RE = re.compile(r"^\s*\[exitCode:\s*-?\d+\]\s*")
_LISTING_ROW_RE = re.compile(r"^[-dl][rwx-]{9}\s", re.M)
_CHECK_HINT_RE = re.compile(
    r"\./(?:check|verify|grade|judge|run)\b"
    r"|(?:^|\s)(?:python3?\s+)?(?:check|verify|grade)\.[a-z]+\b"
)
_PATH_ONLY_RE = re.compile(r"^(?:[A-Za-z]:)?/[\w./+-]+$")
_JSON_START_RE = re.compile(r"[{\[]")
_PATH_OUT_RE = re.compile(r"^PATH:(\S+)", re.M)
_CAT_RE = re.compile(r"\bcat\s+([^\s|;&>()<]+)")
_LOCATE_MARK = "PATH:"
# `-H` 跟着根目录的软链接走（/tmp -> /private/tmp 这类），
# `$(pwd)` 而不是 `.`：命中行必须是绝对路径，沙盒的工作目录每回合可能被重置。
_LOCATE_ROOTS = '/tmp /home /root /opt /srv /data /workspace "$(pwd)"'


def _doc_names(text: str) -> list[str]:
    """任务文本里提到的文件名，优先说明书（.md/.txt）——那才是任务描述。"""
    names = _file_names(text)
    docs = [name for name in names if name.endswith((".md", ".txt"))]
    return docs or names


def _locate_cmd(name: str) -> str:
    """一条命令里同时定位任务描述文件并读出内容。

    判题器每回合只执行一条命令：拆成"先 find 再 cat"要两个回合，而任务超时
    按回合数算、完成回合越早分越高，所以这里合并。常见目录找不到再退回全盘。
    """
    quoted = f"'{name}'"
    return "; ".join((
        f"f=$(find -H {_LOCATE_ROOTS} -name {quoted} 2>/dev/null | head -1)",
        f'[ -n "$f" ] || f=$(find -H / -name {quoted} 2>/dev/null | head -1)',
        f'echo "{_LOCATE_MARK}$f"',
        'cat "$f"',
    ))


def _cat_target(cmd: str) -> str | None:
    """命令里真正被 cat 的文件——模型爱发 `cd X && cat Y` 这种复合命令。"""
    for found in _CAT_RE.finditer(cmd):
        target = found.group(1).strip("\"'")
        if target and "$" not in target:
            return target
    return None


def _strip_exit(result: str) -> str:
    """判题器把沙盒输出包成 ``[exitCode:N]\\n<输出>``，剥掉再判断内容。"""
    return _EXIT_RE.sub("", result or "").strip()


def _is_listing(result: str) -> bool:
    """目录列表 / find 命中行。注意只按"文件名 + 权限位"识别，
    不能拿"以 - 开头"当依据：markdown 的列表项也是 - 开头。"""
    if "drwx" in result or _LISTING_ROW_RE.search(result):
        return True
    lines = [ln for ln in result.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    pathish = sum(1 for ln in lines if ln.lstrip().startswith("/"))
    return pathish > len(lines) * 0.8


def _check_output(body: str) -> bool:
    """验收脚本真打出了东西就算证据——哪怕只是一个很短的 TOKEN 行。

    命令里串了多个备选跑法（``./check || sh ./check || python3 check.py``）时，
    失败的备选会把输出染成"像报错"，所以只要真出现 TOKEN / JSON 就认。
    """
    text = body.strip()
    if not text:
        return False
    if _TOKEN_RE.search(text) or _json_blocks(text):
        return True
    return not _looks_like_error(text)


def _runs_check(cmd: str) -> bool:
    return bool(_CHECK_HINT_RE.search(cmd))


def _json_blocks(text: str) -> list[str]:
    """从任意输出里切出完整、括号配平的 JSON 片段（脚本常把答案打在日志里）。"""
    blocks: list[str] = []
    for start in _JSON_START_RE.finditer(text):
        depth = 0
        in_str = False
        escaped = False
        for index in range(start.start(), min(len(text), start.start() + 4000)):
            char = text[index]
            if in_str:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_str = False
                continue
            if char == '"':
                in_str = True
            elif char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    blocks.append(text[start.start():index + 1])
                    break
    return blocks


def _is_selfevo(text: str) -> bool:
    """自进化类任务：要在沙盒里跑脚本 / 修代码，答案是一次性的（token 等）。"""
    low = text.lower()
    if any(hint in low for hint in _SELFEVO_HINTS):
        return True
    has_token = any(word in low for word in _TOKEN_WORDS)
    has_action = any(hint in low for hint in _ACTION_HINTS)
    return has_token and has_action


def _sandbox_task(session) -> bool:
    """要在沙盒里跑命令的任务：答案必须真跑出来，不能拿半成品交卷。"""
    return _is_selfevo(session.text) or bool(_file_names(session.text))


def _evidence(session) -> bool:
    """交卷门槛：至少真读过一份任务说明，或真跑过一次验收脚本。"""
    return bool(session.read_round or session.ran_round)


def _pending(session) -> list[str]:
    """还没执行过的待办命令。

    跑过的必须从队列里剔掉，否则模型第一问给的瞎猜（上一局就是
    ``["cat task_1_alpha.md"]``）会永远卡在队列里，把后续提问全堵死。
    """
    done = {cmd for cmd, _result in session.transcript}
    session.pending_cmds = [cmd for cmd in session.pending_cmds if cmd not in done]
    return session.pending_cmds


def _at_hunt_stage(session) -> bool:
    """该让模型抽答案了吗：跑过验收脚本，或已经能解析出候选答案。"""
    return bool(session.ran_round) or bool(
        _token_answer(session) or _json_answer(session)
    )


def _workspace(session) -> str | None:
    """工作区目录：优先摘要里给的路径，其次任务描述文件所在的目录。"""
    blob = f"{session.text}\n{session.doc_text}"
    paths = [path.rstrip("/") for path in _PATH_RE.findall(blob)]
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


def _workdir(session) -> str | None:
    return (
        session.workspace
        or _workspace(session)
        or _workspace_from_transcript(session)
    )


def _check_cmd(session) -> str | None:
    """跑验收脚本。一条命令里串上所有常见跑法：脚本没有执行位 / 是 python 时
    也能一次跑通，省下"失败 -> 换个跑法再试"的回合。"""
    ws = _workdir(session)
    if ws is None:
        return None
    return (
        f"cd {ws} && {{ ./check 2>&1 || sh ./check 2>&1 "
        f"|| python3 check.py 2>&1 || true; }} | tail -40"
    )


def _chore(session) -> list[str]:
    """自进化类任务的固定 SOP：看目录 -> 跑验收脚本看报错。"""
    ws = _workdir(session)
    if ws is None:
        return list(_DISCOVER_CMDS)
    check = _check_cmd(session)
    return [
        f"cd {ws} && ls -la && find . -maxdepth 3 -type f "
        f"-not -path './.git/*' | head -40",
        *(item for item in (check,) if item),
    ]


def _unread_docs(session) -> str | None:
    """目录列表/查找结果里出现过的说明文档，逐份读一遍——用绝对路径，
    不能再像上一局那样发 `cat task_1_alpha.md`（cwd 不对必然失败）。

    任务描述已经读到手就不再补读：多读一份 README 要花一个回合，
    而任务分是按完成回合算的。
    """
    ws = _workdir(session)
    if ws is None or session.doc_text:
        return None
    done = {cmd for cmd, _result in session.transcript}
    names: list[str] = []
    for cmd, result in session.transcript:
        if not cmd.startswith(("cd ", "ls ", "find ")):
            continue
        for line in result.splitlines():
            name = line.strip().rsplit(" ", 1)[-1].rsplit("/", 1)[-1]
            if name.endswith((".md", ".txt")) and name not in names:
                names.append(name)
    for name in names:
        cmd = f"cat {ws}/{name}"
        if cmd not in done:
            return cmd
    return None


def _looks_like_error(text: str) -> bool:
    low = text.lower()
    if any(marker in low for marker in _ERROR_MARKERS):
        return True
    head = low[:120]
    return any(word in head for word in _ERROR_HEADS)


def _echoes_task(text: str, session) -> bool:
    """任务原文（tasks.md / spec.md 的内容）不是答案。"""
    body = re.sub(r"\s+", "", text)
    if len(body) < 8:
        return False
    for source in (session.text, session.doc_text):
        target = re.sub(r"\s+", "", source)
        if target and body[:120] in target:
            return True
    return False


def _placeholder_only(text: str) -> bool:
    scrubbed = re.sub(r'"[^"]*"\s*:', "", text)
    scrubbed = re.sub(r"[A-Za-z_]\w*\s*[:=]", "", scrubbed)
    scrubbed = _PLACEHOLDER_RE.sub("", scrubbed)
    scrubbed = re.sub(r"[\s{}\[\]\"',:]+", "", scrubbed)
    return len(scrubbed) < 3


def _plausible(text: str, session) -> bool:
    body = text.strip()
    if not body or len(body) > 2000:
        return False
    if _looks_like_error(body) or _is_listing(body):
        return False
    # 文件路径不是答案：上一局把 find 命中的 task_1_alpha.md 路径当答案交了
    if _PATH_ONLY_RE.match(body) or body in _file_names(session.text):
        return False
    if _echoes_task(body, session) or _placeholder_only(body):
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
        _record(session, session.last_issued, turn.last_cmd_result,
                turn.round_no)
        session.last_issued = ""
    cmd = _next_cmd(turn, state)
    session.last_issued = cmd
    session.last_issued_round = turn.round_no
    return cmd


def _record(session, cmd: str, result: str, round_no: int) -> None:
    result = (result or "")[:_MAX_RESULT]
    session.transcript.append((cmd, result))
    if session.pending_cmds and session.pending_cmds[0] == cmd:
        session.pending_cmds.pop(0)
    if len(session.transcript) > _MAX_TRANSCRIPT:
        session.transcript = session.transcript[-_MAX_TRANSCRIPT:]
    _absorb(session, cmd, result, round_no)


def _absorb(session, cmd: str, result: str, round_no: int) -> None:
    """从命令输出里吸收两类关键状态：任务描述文件在哪、是否真跑出了东西。"""
    body = _strip_exit(result)
    if not body:
        return
    _absorb_locate(session, body, round_no)
    # 1. 单独的 `find <任务里的文件名>` 命中行 -> 记下绝对路径与其所在目录
    if cmd.startswith("find ") and not _looks_like_error(body):
        names = _doc_names(session.text)
        for line in body.splitlines():
            path = line.strip()
            if not path.startswith("/") or path.endswith("/"):
                continue
            if path.rsplit("/", 1)[-1] in names:
                session.task_file = path
                session.workspace = path.rsplit("/", 1)[0]
                break
    # 2. 成功读到一份说明文档 -> 这就是"真读过任务"的证据
    target = _cat_target(cmd)
    if target is not None:
        name = target.rsplit("/", 1)[-1]
        if name.endswith((".md", ".txt")) and not _looks_like_error(body):
            if target == session.task_file or not session.doc_text:
                session.doc_text = body
            session.read_round = session.read_round or round_no
    # 3. 验收脚本真的跑了并有输出
    if _runs_check(cmd) and _check_output(body):
        session.ran_round = session.ran_round or round_no


def _absorb_locate(session, body: str, round_no: int) -> None:
    """解析定位命令：`PATH:<绝对路径>` 一行，其后紧跟文件内容。"""
    match = _PATH_OUT_RE.search(body)
    if match is None:
        return
    path = match.group(1)
    if not path.startswith("/"):
        return
    session.task_file = path
    session.workspace = path.rsplit("/", 1)[0]
    if not path.endswith((".md", ".txt")):
        return
    doc = body[match.end():].strip()
    if not doc or _looks_like_error(doc):
        return
    if not session.doc_text:
        session.doc_text = doc
    session.read_round = session.read_round or round_no


def _next_cmd(turn: Turn, state: GameState) -> str:
    session = state.task
    done = {cmd for cmd, _result in session.transcript}
    # 1. 任务描述文件：一条命令里定位 + 读出内容（判题器每回合只回一条命令的结果）
    if not session.read_round:
        if session.task_file:
            cmd = f"cat {session.task_file}"
            if cmd not in done:
                return cmd
        for name in _doc_names(session.text):
            cmd = _locate_cmd(name)
            if cmd not in done:
                return cmd
    # 3. 工作区：先补读还没读过的说明文档，再走固定剧本
    doc = _unread_docs(session)
    if doc is not None and doc not in done:
        return doc
    for cmd in _chore(session):
        if cmd not in done:
            return cmd
    # 4. 工作区还没定位到才需要全局乱翻；已经知道工作区就别浪费回合了
    if _workdir(session) is None:
        for cmd in _EXPLORE_CMDS:
            if cmd not in done:
                return cmd
    # 5. 大模型给的命令只当兜底：它看不到沙盒结果时只会瞎猜（比如漏掉 cwd）
    for cmd in _pending(session):
        return cmd
    # 6. 验收脚本：换过命令就立刻重跑；否则每 _CHECK_EVERY 回合探一次，
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
    """任务期间向大模型提问：还没跑出东西时要命令，有结果后要答案。"""
    session = state.task
    if not session.active:
        return ""
    _purge_inflight(session, turn.round_no)
    if _pending(session) or session.llm_inflight:
        return ""
    if turn.round_no - session.llm_asked_round < 2:
        return ""
    if session.draft_answer:
        return ""
    # 一条命令都没跑过时提问纯属瞎猜：上一局模型只凭任务文本猜了个
    # `cat task_1_alpha.md`，cwd 不对直接失败。先让确定性流程跑出结果再问。
    if not session.transcript:
        return ""
    session.llm_asked_round = turn.round_no
    # 只有在真要抽答案时才问答案；否则继续要命令（读完说明 ≠ 已经拿到答案）
    if _at_hunt_stage(session):
        session.llm_inflight.append((LLM_EXTRACT, turn.round_no))
        return llm_extract_prompt(state)
    session.llm_inflight.append((LLM_PLAN, turn.round_no))
    return llm_plan_prompt(state)


def apply_llm_resp(turn: Turn, state: GameState, text: str) -> bool:
    """把回来的 llmResp 折进任务状态。

    不严格按提问类型分发：LLM 有时会跳过"给命令"直接给答案，也有时把答案
    包装成命令。先按提问意图处理，再退回去试另一种，两头都不浪费。
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
    if cmds is not None:
        cmds = _sanitize_cmds(cmds, session) or None
    if kind == LLM_EXTRACT and _apply_answer(state, text):
        return True
    if cmds:
        session.pending_cmds = cmds
        session.llm_plan_done = True
        return True
    return _apply_answer(state, text)


def _sanitize_cmds(cmds: list[str], session) -> list[str]:
    """把裸文件名补成绝对路径——模型看不到沙盒结果时只会说 `cat x.md`。"""
    names = _file_names(session.text)
    result: list[str] = []
    for cmd in cmds:
        text = cmd.strip()
        if session.task_file:
            for name in names:
                text = re.sub(
                    rf"(?<![\w/.-]){re.escape(name)}", session.task_file, text,
                )
        if text and text not in result:
            result.append(text)
    return result


def _purge_inflight(session, round_no: int) -> None:
    session.llm_inflight = [
        entry for entry in session.llm_inflight
        if round_no - entry[1] <= _LLM_TTL
    ]


def _as_commands(text: str) -> list[str] | None:
    """只有"看着像 shell 命令的 JSON 数组"才当命令，避免吃掉答案。

    兼容 `["cmd"]` 与 `[{"cmd": "..."}]` 两种写法——模型两种都会给。
    """
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None
    cmds: list[str] = []
    for item in items:
        if isinstance(item, str) and item:
            cmds.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for field in ("cmd", "command", "shell", "run", "exec"):
            value = item.get(field)
            if isinstance(value, str) and value:
                cmds.append(value)
                break
    if not cmds or any(len(cmd) > 500 for cmd in cmds):
        return None
    if not any(_SHELL_HINT_RE.search(cmd) for cmd in cmds):
        return None
    return cmds[:8]


def _apply_answer(state: GameState, text: str) -> bool:
    session = state.task
    if len(text) > 4000:
        return False
    head = text[:80].lower()
    if any(word in head for word in _REFUSAL_WORDS):
        return False
    if text in session.bad_answers or text in session.submitted:
        return False
    # 没真读过任务 / 跑过脚本之前，任何"答案"都是猜的，一律不收
    if not _evidence(session):
        return False
    if session.json_required and not _json_ok(text):
        return False
    if not _plausible(text, session):
        return False
    session.draft_answer = text
    return True


def llm_plan_prompt(state: GameState) -> str:
    session = state.task
    transcript = "\n".join(
        f"$ {cmd}\n{(result or '(无输出)')[:600]}"
        for cmd, result in session.transcript[-6:]
    )
    ws = _workdir(session)
    where = (
        f"工作目录是 {ws}（沙盒每次都会重置），每条命令都要写成 "
        f"cd {ws} && ... 的形式，不要用相对路径猜文件名。\n"
        if ws else "不要猜路径，先用 ls/find 把路径查清楚。\n"
    )
    return (
        "你在一个无网络的 Linux 沙盒里用 shell 命令完成下面的比赛任务。\n"
        f"{where}"
        "命令必须真能在沙盒里跑通；需要看文件就 cat，需要改文件就 sed -i 或重定向。\n"
        f"任务描述：\n{session.text[:1200]}\n"
        f"已执行的命令与输出：\n{transcript or '(尚未执行任何命令)'}\n"
        "请给出下一步要执行的 shell 命令，用于查看文件、修复代码、运行验收脚本，"
        "只输出 JSON 数组，不要解释：[\"cmd1\",\"cmd2\",...]"
    )


def _answer_spec(session) -> str:
    """把"答案该长什么样"写死在提示里：上一局模型直接把文件路径交了上去。"""
    lines: list[str] = []
    if session.json_required:
        lines.append(
            "重要：判题器已回告\"答案不是合法 JSON\"。这次只能输出合法 JSON 本体"
            "（形如 {\"token\": \"...\"}），绝不能输出文件路径、目录列表或解释。"
        )
    key = _JSON_KEY_RE.search(f"{session.text}\n{session.doc_text}")
    if key is not None:
        name = key.group(1)
        lines.append(
            f"任务要求 JSON，键名是 \"{name}\"，请严格输出 {{\"{name}\": \"<值>\"}}。"
            f"<值>必须是沙盒里真跑出来的结果，不能是任务描述里的占位符。"
        )
    if session.last_error:
        lines.append(f"判题器最近一次反馈：{session.last_error}")
    return "".join(line + "\n" for line in lines)


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
        "你需要提交一个比赛任务的最终答案。\n"
        f"任务描述：\n{session.text[:1200]}\n"
        "沙盒执行记录（这是唯一可信的信息来源）：\n"
        f"{transcript}\n"
        f"{rejected}"
        f"{_answer_spec(session)}"
        "只输出答案本体，不要解释、不要复述任务原文、不要输出命令、"
        "不要输出文件路径或目录列表。"
    )


def _json_ok(text: str) -> bool:
    try:
        json.loads(text)
    except Exception:
        return False
    return True


def _resolve_answer(session) -> str:
    answer = _token_answer(session)
    if answer:
        return answer
    answer = _json_answer(session)
    if answer:
        return answer
    if session.json_required or _sandbox_task(session):
        return ""  # 沙盒任务必须真跑出结果，不能拿半成品交卷
    return _text_answer(session)


def _json_answer(session) -> str:
    """沙盒脚本常把答案当 JSON 打在输出里，取最后一段能解析的。

    输出里混了失败备选的报错也不能整段丢掉：只要切出了 JSON 片段就照常解析，
    是不是答案交给 ``_json_candidate`` 把关。
    """
    for _cmd, result in reversed(session.transcript):
        if _echoes_task(result, session):
            continue
        body = _strip_exit(result)
        blocks = _json_blocks(body)
        if not blocks and _looks_like_error(body):
            continue
        for block in reversed(blocks):
            answer = _json_candidate(block, session)
            if answer:
                return answer
    return ""


def _json_candidate(block: str, session) -> str:
    try:
        payload = json.loads(block)
    except Exception:
        return ""
    if not isinstance(payload, (dict, list)) or not payload:
        return ""
    answer = json.dumps(payload, ensure_ascii=False)
    # 任务说明里常写"答案格式：{"token": "<TOKEN>"}"，那是模板不是答案
    if answer in session.bad_answers or _PATH_ONLY_RE.match(answer):
        return ""
    if _placeholder_only(answer) or _echoes_task(answer, session):
        return ""
    return answer


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
        if _echoes_task(result, session):
            continue
        for pattern in patterns:
            for found in pattern.finditer(result):
                value = found.group(1)
                if value.lower() in _PLACEHOLDERS or len(value) < 2:
                    continue
                if _PATH_ONLY_RE.match(value) or value in _file_names(session.text):
                    continue  # 文件路径不是答案（上一局就是这么交错的）
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
        if _plausible(body, session):
            return body
    return ""


# ---------------------------------------------------------------- 交卷


def answer_ready(turn: Turn, state: GameState) -> bool:
    """有可信答案就交——判题器按"历史最高通过率"记分，早交不会更差。"""
    session = state.task
    if not session.active or session.step != "EXPLORING":
        return False
    # 证据门槛：一次沙盒命令都没真跑出东西之前，绝不交卷
    if not _evidence(session):
        return False
    answer = session.draft_answer or _resolve_answer(session)
    if not answer:
        return False
    if session.json_required and not _json_ok(answer):
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
    for code, desc in turn.errors:
        if code != 2:
            continue
        # 判题器会直说"答案不是合法 JSON"——记下来，后面只收 JSON
        session.last_error = desc
        if "json" in desc.lower():
            session.json_required = True
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
    # 沙盒任务的答案每次都不一样，只复用命令流程
    if not _sandbox_task(session):
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
