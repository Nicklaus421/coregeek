"""几何与寻路工具。

- 距离统一使用切比雪夫距离（任务书 4.5.4）。
- 移动为 8 方向，每回合走一格。
"""
from __future__ import annotations

from collections import deque

from .models import Pos

DIRS8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def cheb(a: Pos, b: Pos) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


def cheb_xy(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(ax - bx), abs(ay - by))


def in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def neighbors(p: Pos, w: int, h: int):
    for dx, dy in DIRS8:
        nx, ny = p.x + dx, p.y + dy
        if in_bounds(nx, ny, w, h):
            yield Pos(nx, ny)


def bfs_path(start: Pos, goal: Pos, blocked: set[tuple[int, int]], w: int, h: int) -> list[Pos] | None:
    """8 方向 BFS，返回从 start 到 goal 的路径（含 goal、不含 start）。

    ``goal`` 若在 blocked 中仍视为可达终点（移动到目标旁边时目标本身常被占据）。
    无路可达返回 None。
    """
    if start.key() == goal.key():
        return []
    blocked = set(blocked)
    blocked.discard(goal.key())
    if start.key() in blocked:
        return None

    q: deque[Pos] = deque([start])
    came: dict[tuple[int, int], Pos | None] = {start.key(): None}
    while q:
        cur = q.popleft()
        if cur.key() == goal.key():
            break
        for n in neighbors(cur, w, h):
            k = n.key()
            if k in came or k in blocked:
                continue
            came[k] = cur
            q.append(n)

    if goal.key() not in came:
        return None

    path: list[Pos] = []
    cur: Pos | None = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur.key()]
    path.reverse()
    return path[1:]  # 去掉 start


def ring_cells(base_cells: list[Pos], radius: int, w: int, h: int) -> list[Pos]:
    """返回与 base 块的切比雪夫距离恰好为 ``radius`` 的格子（边界内），按 (x, y) 排序。

    8 方向 BFS 的步数即切比雪夫距离，因此可用 BFS 计算距离层。
    """
    dist: dict[tuple[int, int], int] = {}
    q: deque[Pos] = deque()
    for c in base_cells:
        dist[c.key()] = 0
        q.append(c)
    while q:
        c = q.popleft()
        d = dist[c.key()]
        for n in neighbors(c, w, h):
            k = n.key()
            if k not in dist:
                dist[k] = d + 1
                q.append(n)
    result = [Pos(x, y) for (x, y), d in dist.items() if d == radius]
    result.sort(key=lambda p: (p.x, p.y))
    return result
