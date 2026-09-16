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


def step_adjacent(turn: Turn, moving: Unit, goal: Pos) -> Pos | None:
    """走向 goal 周围一格的可达空地；已经相邻则返回 None。"""
    if distance(moving.pos, goal) <= 1 and moving.pos != goal:
        return None
    blocked = turn.blocked(moving)
    stands = [
        Pos(goal.x + dx, goal.y + dy) for dx, dy in _STEPS
    ]
    stands = [
        pos for pos in stands
        if pos == moving.pos or (turn.land(pos) and pos not in blocked)
    ]
    stands.sort(key=lambda pos: distance(pos, moving.pos))
    for stand in stands:
        if stand == moving.pos:
            return None
        step = next_step(turn, moving, stand)
        if step is not None:
            return step
    return None


def _first_step(came_from: dict[Pos, Pos], start: Pos, goal: Pos) -> Pos:
    current = goal
    while came_from[current] != start:
        current = came_from[current]
    return current
