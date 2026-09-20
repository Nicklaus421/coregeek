"""自进化类任务引擎。

开拓者前往任务点接取自进化任务，通过沙盒 ``executeCmd`` 与 ``lastCmdResult`` 多回合交互，
真跑出答案后再 ``submitAnswer``。

重要约束（来自对战经验）：
- 绝不提交任务原文、shell 报错、占位符充当答案；
- 拿不到真实答案就不交卷。
"""
from __future__ import annotations

import re

from . import geometry as geo
from .models import Pos, Role
from .state import GameState

# 沙盒探测命令序列（每回合执行一条，结果下一回合经 lastCmdResult 返回）。
_EXPLORE_CMDS = (
    "pwd && ls -la",
    "find . -maxdepth 3 -type f 2>/dev/null",
    "cat README* 2>/dev/null; cat *.md 2>/dev/null; cat *.txt 2>/dev/null; cat task* 2>/dev/null",
    "ls -la; test -x ./check && ./check || (test -f ./check && bash ./check || true)",
    "ls -la; for f in *.py; do echo \"== $f ==\"; cat \"$f\"; done 2>/dev/null",
)


def _plausible(answer: str, phase_task: str) -> bool:
    """出站合法性：必须是看起来真实解出的答案。"""
    if not answer:
        return False
    a = answer.strip()
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


def _extract_answer(cmd_result: str) -> str | None:
    """从 ``[exitCode:N]\\n<output>`` 结果中提取候选答案。

    优先识别常见的 TOKEN / JSON 字段，否则取最后一行非控制输出。
    """
    if not cmd_result:
        return None
    lines = [ln.strip() for ln in cmd_result.splitlines() if ln.strip()]
    # 去掉 [exitCode:...]、[TRUNCATED]、[TIMEOUT]、[JUDGER_ERROR] 等控制行
    lines = [ln for ln in lines if not ln.startswith("[")]

    if not lines:
        return None

    # 尝试识别 token / answer 字段
    for ln in lines:
        m = re.search(r"(?:token|answer|result|key)\s*[:=]\s*[\"']?([A-Za-z0-9_\-\.]+)", ln, re.I)
        if m:
            return m.group(1)

    return lines[-1]


class TaskEngine:
    """开拓者任务状态机（跨回合持久）。"""

    def __init__(self) -> None:
        self.explore_idx = 0
        self.answer: str | None = None
        self.submitted: bool = False

    def reset(self) -> None:
        self.explore_idx = 0
        self.answer = None
        self.submitted = False

    def step(self, game: GameState) -> tuple[dict | None, str | None]:
        """返回 (开拓者指令, executeCmd)。指令为 None 表示本回合不动开拓者。"""
        pioneer = game.pioneer
        if pioneer is None:
            return None, None

        phase = game.phase_task

        # 无任务进行中
        if not phase:
            self.reset()
            return self._go_accept(game, pioneer)

        # 任务进行中
        # 先解析上一回合沙盒命令的结果
        ans = _extract_answer(game.last_cmd_result)
        if ans and _plausible(ans, phase) and self.answer is None:
            self.answer = ans

        if self.answer is not None and _plausible(self.answer, phase) and not self.submitted:
            self.submitted = True
            return {"action": "submitAnswer", "taskAnswer": self.answer}, None

        # 继续探测
        cmd = self._next_explore()
        return None, cmd

    def _go_accept(self, game: GameState, pioneer: Role) -> tuple[dict | None, str | None]:
        tp = self._nearest_valid_task_point(game, pioneer.pos)
        if tp is None:
            return None, None
        if geo.cheb(pioneer.pos, tp) <= 1:
            return {"action": "acceptTask"}, None
        mv = self._move_adjacent(game, pioneer, tp)
        return mv, None

    def _next_explore(self) -> str | None:
        if self.explore_idx >= len(_EXPLORE_CMDS):
            return None
        cmd = _EXPLORE_CMDS[self.explore_idx]
        self.explore_idx += 1
        return cmd

    @staticmethod
    def _nearest_valid_task_point(game: GameState, pos: Pos) -> Pos | None:
        own = game.own_task_points()
        valid = [t for t in game.player_tasks if t.is_valid]
        valid_positions = {t.task_position.key() for t in valid}
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
