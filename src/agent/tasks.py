"""自进化类任务引擎（LLM 驱动）。

与判题器的完整交互闭环：

1. 开拓者 ``acceptTask`` 领取任务。
2. 下一回合 ``req.phaseTask`` 描述任务（通常是"读取 xxx.md 文档"）。
3. 用一条 ``executeCmd``（``find -H`` 多路径定位 + 兜底全盘 ``find`` + ``cat``）直接读出文档内容，结果经下一回合 ``lastCmdResult`` 返回；此步不与大模型交互。
4. 把任务描述 + 已收集信息通过 ``resp.prompt`` 交给判题器大模型，结果经下一回合 ``llmResp`` 返回。
5. 根据 ``llmResp`` 决定：执行命令（``executeCmd``）或提交答案（``submitAnswer``）。
6. 命令结果逐回合累积，循环 prompt → 命令/答案，直到得到最终答案提交。

约束：绝不提交任务原文、shell 报错、占位符充当答案；拿不到真实答案就不交卷。
"""
from __future__ import annotations

import difflib
import re

from . import geometry as geo
from .models import Pos, Role
from .state import GameState

_PROMPT_TEMPLATE = """你是编码大赛里的一个 Agent，需要通过沙盒环境完成一个任务。

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
    READ_DOC = "read_doc"      # 下一动作：一条 find+cat 命令直接读出文档内容
    WAIT_DOC = "wait_doc"      # 已发读文档命令，等 lastCmdResult（文件内容）
    WAIT_LLM = "wait_llm"      # 已发 prompt，等 llmResp
    WAIT_CMD = "wait_cmd"      # 已发 executeCmd，等 lastCmdResult
    REPLAY = "replay"          # 复用已记录的 SOP：逐回合重放命令，无需重新探索
    DONE = "done"              # 已提交答案

    def __init__(self) -> None:
        self.mode = self.IDLE
        self.task_desc = ""
        self.task_type = ""
        self.history: list[tuple[str, str]] = []
        self.pending_cmd: str | None = None
        self.answer: str | None = None
        # SOP 缓存：task_type -> {"commands": [...], "task": 首次任务原文}
        # 首次成功解出某类任务后记录，后续同类任务直接重放命令（可跨回合/复活持久）。
        self.sop: dict[str, dict] = {}
        # 当前任务首次求解中已发出的命令序列（用于成功后固化为 SOP）
        self._recorded_cmds: list[str] = []
        # 重放态：待重放的命令序列与进度
        self._replay_cmds: list[str] = []
        self._replay_idx = 0

    def reset(self) -> None:
        """清空当前任务的瞬时状态，但保留已固化的 SOP 缓存。"""
        self.mode = self.IDLE
        self.task_desc = ""
        self.task_type = ""
        self.history = []
        self.pending_cmd = None
        self.answer = None
        self._recorded_cmds = []
        self._replay_cmds = []
        self._replay_idx = 0

    def step(self, game: GameState) -> tuple[dict | None, str | None, str]:
        """返回 (开拓者指令, executeCmd, prompt)。"""
        pioneer = game.pioneer
        if pioneer is None:
            return None, None, ""
        phase = game.phase_task

        # 无任务进行中：提交上一任务 SOP（若成功），前往接取
        if not phase:
            self._commit_sop()
            self.reset()
            cmd, _, _ = self._go_accept(game, pioneer)
            return cmd, None, ""

        # 新任务开始
        if phase != self.task_desc:
            self._commit_sop()
            self.task_type = self._current_task_type(game, pioneer)
            self.task_desc = phase
            self.history = []
            self.pending_cmd = None
            self.answer = None
            self._recorded_cmds = []
            self._replay_cmds = []
            self._replay_idx = 0
            if self.task_type in self.sop:
                # 已存在 SOP：直接重放（命令按新任务原文做参数替换）
                self._replay_cmds = self._substitute_commands(
                    self.sop[self.task_type]["commands"],
                    self.sop[self.task_type]["task"],
                    phase,
                )
                self.mode = self.REPLAY
            else:
                self.mode = self.READ_DOC

        # 累积本回合回传的结果
        if self.mode == self.WAIT_DOC and game.last_cmd_result:
            self.history.append((self.pending_cmd or "读取文档", game.last_cmd_result))
            self.pending_cmd = None
        elif self.mode == self.WAIT_CMD and game.last_cmd_result:
            self.history.append((self.pending_cmd or "", game.last_cmd_result))
            self.pending_cmd = None
        elif self.mode == self.REPLAY and game.last_cmd_result:
            self.history.append((self.pending_cmd or "", game.last_cmd_result))
            self.pending_cmd = None
        elif self.mode == self.WAIT_LLM:
            pass  # llmResp 在 _advance 里解析

        return self._advance(game, pioneer)

    def _advance(self, game: GameState, pioneer: Role) -> tuple[dict | None, str | None, str]:
        phase = game.phase_task

        if self.mode == self.READ_DOC:
            fname = _extract_md_filename(phase)
            if fname is None:
                # 无明确文件名，直接兜底读取（只输出内容）
                cmd = self._glob_read_cmd()
            else:
                cmd = self._read_doc_cmd(fname)
            self.pending_cmd = cmd
            self._recorded_cmds.append(cmd)
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
                self._recorded_cmds.append(cmd)
                self.mode = self.WAIT_CMD
                return None, cmd, ""
            # 大模型未给出可执行指令，重新询问
            return self._send_prompt()

        if self.mode == self.WAIT_CMD:
            # 命令结果已回（在 step 中已累积），交给大模型
            return self._send_prompt()

        if self.mode == self.REPLAY:
            # 逐回合重放已记录的 SOP 命令，全部放完后再用一次 prompt 收敛答案
            if self._replay_idx < len(self._replay_cmds):
                cmd = self._replay_cmds[self._replay_idx]
                self._replay_idx += 1
                self.pending_cmd = cmd
                return None, cmd, ""
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
    def _commit_sop(self) -> None:
        """首次成功解出某类任务后，把命令序列固化为该类型的 SOP。

        仅记录首次（``task_type`` 尚不在 ``sop`` 中）；后续重放成功不会覆盖，
        避免把「重放 + 一次 prompt」的过程误当成全新探索流程写回。
        """
        if (
            self.task_type
            and self.task_type not in self.sop
            and self.answer is not None
            and self._recorded_cmds
        ):
            self.sop[self.task_type] = {
                "commands": list(self._recorded_cmds),
                "task": self.task_desc,
            }

    @staticmethod
    def _current_task_type(game: GameState, pioneer: Role) -> str:
        """推断开拓者当前接取的是哪类自进化任务。

        任务执行期间开拓者须停留在任务点一格内，据此用 ``player_tasks`` 里与
        开拓者相邻的任务点匹配 ``taskType``；匹配不到则回退到最近的自进化任务点。
        """
        se_types = ("自进化类1", "自进化类2")
        se = [t for t in game.player_tasks if t.task_type in se_types]
        for t in se:
            if geo.cheb(t.task_position, pioneer.pos) <= 1:
                return t.task_type
        if not se:
            return ""
        return min(se, key=lambda t: geo.cheb(t.task_position, pioneer.pos)).task_type

    @staticmethod
    def _substitute_commands(commands: list[str], old_task: str, new_task: str) -> list[str]:
        """把 SOP 命令里与旧任务绑定的参数替换成新任务的参数。

        用 difflib 定位旧任务原文与新任务原文的差异片段（如「北京」→「上海」），
        再将差异片段在每条命令中做字符串替换，使重放的命令作用于新任务目标。
        """
        if not old_task or old_task == new_task:
            return list(commands)
        subs: list[tuple[str, str]] = []
        sm = difflib.SequenceMatcher(a=old_task, b=new_task)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "replace":
                subs.append((old_task[i1:i2], new_task[j1:j2]))
        out: list[str] = []
        for c in commands:
            for old, new in subs:
                c = c.replace(old, new)
            out.append(c)
        return out

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
    def _read_doc_cmd(fname: str) -> str:
        """一条命令直接读出文档内容：多路径定位 + 兜底全盘查找 + cat。

        先在常见目录（含当前工作目录）里用 ``find -H`` 找文件名，找不到再全盘
        ``find -H /`` 兜底，命中后 ``cat`` 输出内容；全程抑制 stderr，避免
        ``permission denied`` 等中间产物混入 ``lastCmdResult``。
        """
        base = fname.rsplit("/", 1)[-1]
        return (
            'f=$(find -H /tmp /home /root /opt /srv /data /workspace "$(pwd)" '
            f"-name '{base}' 2>/dev/null | head -1); "
            f"[ -n \"$f\" ] || f=$(find -H / -name '{base}' 2>/dev/null | head -1); "
            'cat "$f"'
        )

    @staticmethod
    def _glob_read_cmd() -> str:
        """兜底：无明确文件名时，只输出 CWD 下 md/txt 内容，不带 ls/pwd 等中间产物。"""
        return 'for f in *.md *.txt; do [ -f "$f" ] && cat "$f"; done 2>/dev/null'

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
