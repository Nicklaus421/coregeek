"""自进化类任务引擎（LLM 驱动）。

与判题器的完整交互闭环：

1. 开拓者 ``acceptTask`` 领取任务。
2. 下一回合 ``req.phaseTask`` 描述任务（通常是"读取 xxx.md 文档"）。
3. 先用 ``executeCmd`` 定位 xxx.md 的真实路径（``find`` 只打印路径、抑制 stderr），再用 ``executeCmd`` ``cat`` 该路径读取内容，结果均经下一回合 ``lastCmdResult`` 返回。
4. 把任务描述 + 已收集信息通过 ``resp.prompt`` 交给判题器大模型，结果经下一回合 ``llmResp`` 返回。
5. 根据 ``llmResp`` 决定：执行命令（``executeCmd``）或提交答案（``submitAnswer``）。
6. 命令结果逐回合累积，循环 prompt → 命令/答案，直到得到最终答案提交。

约束：绝不提交任务原文、shell 报错、占位符充当答案；拿不到真实答案就不交卷。
"""
from __future__ import annotations

import re

from . import geometry as geo
from .models import Pos, Role
from .state import GameState

_PROMPT_TEMPLATE = """你是《未来战争》游戏里的一个 Agent，需要通过沙盒环境完成一个任务。

请根据下面的【任务】与【已收集信息】决定下一步操作，并严格按以下格式回复（二选一，不要输出其他任何内容）：

需要执行 shell 命令时：
ACTION=CMD
COMMAND=<要执行的命令>

已经得到最终答案时：
ACTION=ANSWER
ANSWER=<最终答案>

【任务】
{task}

【已收集信息】
{context}
"""

_CMD_PREFIXES = (
    "ls", "cat", "cd", "find", "grep", "python", "echo", "./", "bash", "sh ",
    "pwd", "head", "tail", "sed", "awk", "test", "which", "for ", "if ",
    "mkdir", "touch", "curl", "wget", "pip",
)


def _plausible(answer: str, phase_task: str) -> bool:
    """出站合法性：必须是看起来真实解出的答案，而非任务原文/报错/占位符。"""
    if not answer:
        return False
    a = answer.strip().strip('"').strip("'").strip()
    if not a:
        return False
    if a == phase_task.strip():
        return False
    low = a.lower()
    if "no such file" in low or "command not found" in low or "not found" in low:
        return False
    if a in ("unknown", "null", "none", "{}", "[]", "true", "false"):
        return False
    if a.startswith("error") or a.startswith("traceback"):
        return False
    return True


def _extract_md_filename(phase_task: str) -> str | None:
    m = re.search(r"[A-Za-z_][\w\-\./]+\.md", phase_task)
    return m.group(0) if m else None


def _looks_like_command(text: str) -> bool:
    t = text.strip().strip("`").strip()
    if not t:
        return False
    first = t.splitlines()[0].strip()
    return any(first.startswith(p) for p in _CMD_PREFIXES)


def _parse_llm(text: str) -> tuple[str | None, str | None]:
    """解析大模型回复，返回 (command, answer)，二者其一非 None。"""
    if not text:
        return None, None
    t = text.strip()

    # 1. 我们 prompt 里要求的显式 ACTION 标记
    if re.search(r"ACTION\s*=\s*CMD", t, re.I):
        m = re.search(r"COMMAND\s*=\s*(.+?)(?=\n\s*\n|\Z)", t, re.I | re.S)
        return (m.group(1).strip() if m else None), None
    if re.search(r"ACTION\s*=\s*ANSWER", t, re.I):
        m = re.search(r"ANSWER\s*=\s*(.+?)(?=\n\s*\n|\Z)", t, re.I | re.S)
        return None, (m.group(1).strip() if m else None)

    # 2. 常见答案标记
    m = re.search(r"(?:最终答案|答案)\s*[:：]\s*(.+)", t, re.I | re.S)
    if m:
        return None, m.group(1).strip()
    m = re.search(r"ANSWER\s*[:=]\s*(.+)", t, re.I | re.S)
    if m:
        return None, m.group(1).strip()

    # 3. 常见命令标记
    m = re.search(r"(?:COMMAND|CMD|命令)\s*[:=：]\s*(.+)", t, re.I | re.S)
    if m:
        return m.group(1).strip(), None

    # 4. 启发式回退：像命令 → 命令，否则 → 答案
    if _looks_like_command(t):
        return t, None
    return None, t


class TaskEngine:
    """开拓者任务状态机（跨回合持久）。"""

    IDLE = "idle"
    FIND_FILE = "find_file"    # 下一动作：定位文件路径（find，只打印路径）
    WAIT_FIND = "wait_find"    # 已发 find，等 lastCmdResult（文件路径）
    WAIT_DOC = "wait_doc"      # 已发 cat，等 lastCmdResult（文件内容）
    WAIT_LLM = "wait_llm"      # 已发 prompt，等 llmResp
    WAIT_CMD = "wait_cmd"      # 已发 executeCmd，等 lastCmdResult
    DONE = "done"              # 已提交答案

    def __init__(self) -> None:
        self.mode = self.IDLE
        self.task_desc = ""
        self.history: list[tuple[str, str]] = []
        self.pending_cmd: str | None = None
        self.doc_path: str | None = None
        self.answer: str | None = None

    def reset(self) -> None:
        self.mode = self.IDLE
        self.task_desc = ""
        self.history = []
        self.pending_cmd = None
        self.doc_path = None
        self.answer = None

    def step(self, game: GameState) -> tuple[dict | None, str | None, str]:
        """返回 (开拓者指令, executeCmd, prompt)。"""
        pioneer = game.pioneer
        if pioneer is None:
            return None, None, ""
        phase = game.phase_task

        # 无任务进行中：前往接取
        if not phase:
            self.reset()
            cmd, _, _ = self._go_accept(game, pioneer)
            return cmd, None, ""

        # 新任务开始
        if phase != self.task_desc:
            self.task_desc = phase
            self.history = []
            self.pending_cmd = None
            self.doc_path = None
            self.answer = None
            self.mode = self.FIND_FILE

        # 累积本回合回传的结果
        if self.mode == self.WAIT_FIND:
            # find 只回文件路径，内部消费，不进历史
            self.doc_path = self._parse_path(game.last_cmd_result)
        elif self.mode == self.WAIT_DOC and game.last_cmd_result:
            self.history.append((self.pending_cmd or "读取文档", game.last_cmd_result))
            self.pending_cmd = None
        elif self.mode == self.WAIT_CMD and game.last_cmd_result:
            self.history.append((self.pending_cmd or "", game.last_cmd_result))
            self.pending_cmd = None
        elif self.mode == self.WAIT_LLM:
            pass  # llmResp 在 _advance 里解析

        return self._advance(game, pioneer)

    def _advance(self, game: GameState, pioneer: Role) -> tuple[dict | None, str | None, str]:
        phase = game.phase_task

        if self.mode == self.FIND_FILE:
            fname = _extract_md_filename(phase)
            if fname is None:
                # 无明确文件名，直接兜底读取（只输出内容）
                cmd = self._glob_read_cmd()
                self.pending_cmd = cmd
                self.mode = self.WAIT_DOC
                return None, cmd, ""
            cmd = self._find_cmd(fname)
            self.pending_cmd = cmd
            self.mode = self.WAIT_FIND
            return None, cmd, ""

        if self.mode == self.WAIT_FIND:
            # 已定位到路径（在 step 中解析），再读取内容
            if self.doc_path:
                cmd = self._read_cmd(self.doc_path)
            else:
                cmd = self._glob_read_cmd()
            self.pending_cmd = cmd
            self.mode = self.WAIT_DOC
            return None, cmd, ""

        if self.mode == self.WAIT_DOC:
            # 文档内容已回，交给大模型
            return self._send_prompt()

        if self.mode == self.WAIT_LLM:
            cmd, ans = _parse_llm(game.llm_resp)
            if ans is not None and _plausible(ans, phase):
                self.answer = ans
                self.mode = self.DONE
                return {"action": "submitAnswer", "taskAnswer": ans}, None, ""
            if cmd is not None:
                self.pending_cmd = cmd
                self.mode = self.WAIT_CMD
                return None, cmd, ""
            # 大模型未给出可执行指令，重新询问
            return self._send_prompt()

        if self.mode == self.WAIT_CMD:
            # 命令结果已回（在 step 中已累积），交给大模型
            return self._send_prompt()

        if self.mode == self.DONE:
            # 已提交；若被判答案错误且任务仍在，重试
            if phase == self.task_desc and self._has_answer_error(game):
                self.history.append(("提交结果", "答案被判错误，请重新推理后再提交"))
                self.answer = None
                return self._send_prompt()
            return None, None, ""

        return None, None, ""

    # ---- 辅助 ----
    def _send_prompt(self) -> tuple[None, None, str]:
        self.mode = self.WAIT_LLM
        return None, None, self._build_prompt()

    def _build_prompt(self) -> str:
        lines = []
        for i, (cmd, result) in enumerate(self.history, 1):
            lines.append(f"[{i}] 命令：{cmd}\n结果：{result}")
        context = "\n\n".join(lines) if lines else "（暂无）"
        return _PROMPT_TEMPLATE.format(task=self.task_desc, context=context)

    @staticmethod
    def _find_cmd(fname: str) -> str:
        """定位文件路径：只打印路径，stderr 抑制，找到即停，避免中间产物污染结果。"""
        base = fname.rsplit("/", 1)[-1]
        return f"find . -name '{base}' -print -quit 2>/dev/null"

    @staticmethod
    def _read_cmd(path: str) -> str:
        """读取已定位的文件内容，只输出内容。"""
        return f"cat '{path}' 2>/dev/null"

    @staticmethod
    def _glob_read_cmd() -> str:
        """兜底：无明确文件名时，只输出 CWD 下 md/txt 内容，不带 ls/pwd 等中间产物。"""
        return 'for f in *.md *.txt; do [ -f "$f" ] && cat "$f"; done 2>/dev/null'

    @staticmethod
    def _parse_path(result: str) -> str | None:
        """从 find 输出里提取文件路径，过滤 permission denied 等干扰行。"""
        if not result:
            return None
        for ln in result.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("["):
                continue
            if "permission denied" in ln.lower() or ln.startswith("find:"):
                continue
            return ln
        return None

    @staticmethod
    def _has_answer_error(game: GameState) -> bool:
        return any(e.get("errorCode") == 2 for e in game.errors)

    def _go_accept(self, game: GameState, pioneer: Role) -> tuple[dict | None, None, str]:
        tp = self._nearest_valid_task_point(game, pioneer.pos)
        if tp is None:
            return None, None, ""
        if geo.cheb(pioneer.pos, tp) <= 1:
            return {"action": "acceptTask"}, None, ""
        return self._move_adjacent(game, pioneer, tp), None, ""

    @staticmethod
    def _nearest_valid_task_point(game: GameState, pos: Pos) -> Pos | None:
        own = game.own_task_points()
        valid_positions = {t.task_position.key() for t in game.player_tasks if t.is_valid}
        candidates = [z.pos for z in own if z.pos.key() in valid_positions]
        if not candidates:
            return None
        return min(candidates, key=lambda p: geo.cheb(pos, p))

    @staticmethod
    def _move_adjacent(game: GameState, role: Role, target: Pos) -> dict | None:
        if geo.cheb(role.pos, target) <= 1:
            return None
        blocked = game.blocked_set(exclude_role_id=role.id)
        best: list[Pos] | None = None
        for n in geo.neighbors(target, game.width, game.height):
            if n.key() in blocked:
                continue
            path = geo.bfs_path(role.pos, n, blocked, game.width, game.height)
            if path and (best is None or len(path) < len(best)):
                best = path
        if best:
            return {"action": "move", "targetPos": [best[0].to_dict()]}
        return None
