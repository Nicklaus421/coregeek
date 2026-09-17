from heapq import heappop, heappush
from itertools import count

from .protocol import Pos, Turn, Unit, distance

_STEPS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def next_step(
    turn: Turn,
    moving: Unit,
    goal: Pos,
    max_expansions: int = 2000,
) -> Pos | None:
    blocked = turn.blocked(moving)
    order = count()
    frontier: list[tuple[int, int, int, Pos]] = [
        (distance(moving.pos, goal), 0, next(order), moving.pos)
    ]
    came_from: dict[Pos, Pos] = {}
    best = {moving.pos: 0}
    seen: set[Pos] = set()

    while frontier and len(seen) < max_expansions:
        _, cost, _, current = heappop(frontier)
        if current in seen:
            continue
        if current == goal:
            return _first_step(came_from, moving.pos, goal)
        seen.add(current)
        for dx, dy in _STEPS:
            step = Pos(current.x + dx, current.y + dy)
            if step in blocked or not turn.land(step):
                continue
            new_cost = cost + 1
            if new_cost >= best.get(step, new_cost + 1):
                continue
            best[step] = new_cost
            came_from[step] = current
            heappush(
                frontier,
                (
                    new_cost + distance(step, goal),
                    new_cost,
                    next(order),
                    step,
                ),
            )
    return None


def step_adjacent(
    turn: Turn,
    moving: Unit,
    goal: Pos,
    max_expansions: int = 2000,
) -> Pos | None:
    """走向 goal 周围一格的可达空地；已经相邻则返回 None。

    单次 A*（终点判定放宽为“进入 goal 相邻的一格”），
    比逐个候选落脚点各跑一次 A* 省一个数量级的扩展。
    """
    if distance(moving.pos, goal) <= 1 and moving.pos != goal:
        return None
    blocked = turn.blocked(moving)
    order = count()
    start = moving.pos
    frontier: list[tuple[int, int, int, Pos]] = [
        (max(0, distance(start, goal) - 1), 0, next(order), start)
    ]
    came_from: dict[Pos, Pos] = {}
    best = {start: 0}
    seen: set[Pos] = set()

    while frontier and len(seen) < max_expansions:
        _, cost, _, current = heappop(frontier)
        if current in seen:
            continue
        if current != start and distance(current, goal) <= 1:
            return _first_step(came_from, start, current)
        seen.add(current)
        for dx, dy in _STEPS:
            step = Pos(current.x + dx, current.y + dy)
            if step in blocked or not turn.land(step):
                continue
            new_cost = cost + 1
            if new_cost >= best.get(step, new_cost + 1):
                continue
            best[step] = new_cost
            came_from[step] = current
            heappush(
                frontier,
                (
                    new_cost + max(0, distance(step, goal) - 1),
                    new_cost,
                    next(order),
                    step,
                ),
            )
    return None


def _first_step(came_from: dict[Pos, Pos], start: Pos, goal: Pos) -> Pos:
    current = goal
    while came_from[current] != start:
        current = came_from[current]
    return current
