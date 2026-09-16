from .grid import next_step, step_adjacent
from .protocol import (
    GATLING,
    RAILGUN,
    ROCKET,
    Pos,
    Robot,
    Turn,
    Unit,
    WALL,
    attack_command,
    distance,
    move_command,
    use_command,
)
from .state import GameState
from .validate import cone_ok

_ROCKET_CENTER_DMG = 20
_ROCKET_SPLASH_DMG = 10
_GATLING_DMG = 10
_ROCKET_MIN_GAIN = 30
_WALL_FIX_RATIO = 0.4


def night_commands(turn: Turn, state: GameState) -> dict[int, dict]:
    commands: dict[int, dict] = {}
    claimed: set[Pos] = set()
    pairs = assign_controllers(turn)
    for role, tower in pairs:
        if distance(role.pos, tower.pos) > 1:
            step = step_adjacent(turn, role, tower.pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                commands[role.unit_id] = move_command(step)
            continue
        if tower.cooldown > 0:
            continue
        targets = _targets_for(turn, tower)
        if targets:
            commands[tower.unit_id] = attack_command(role.unit_id, targets)
    _consumables(turn, pairs, commands)
    return commands


def assign_controllers(turn: Turn) -> tuple[tuple[Unit, Unit], ...]:
    roles = list(turn.controllable())
    towers = list(turn.weapons())
    pairs: list[tuple[Unit, Unit]] = []
    used: set[int] = set()
    options = sorted(
        (
            (distance(role.pos, tower.pos), role.unit_id, tower.unit_id)
            for role in roles
            for tower in towers
        )
    )
    for _dist, role_id, tower_id in options:
        if role_id in used or tower_id in used:
            continue
        used.add(role_id)
        used.add(tower_id)
        role = next(r for r in roles if r.unit_id == role_id)
        tower = next(t for t in towers if t.unit_id == tower_id)
        pairs.append((role, tower))
    return tuple(pairs)


def _targets_for(turn: Turn, tower: Unit) -> list[Pos]:
    if tower.kind == GATLING:
        return gatling_targets(turn, tower)
    if tower.kind == RAILGUN:
        target = railgun_target(turn, tower)
        return [target] if target is not None else []
    if tower.kind == ROCKET:
        return rocket_targets(turn, tower)
    return []


def _candidates(turn: Turn, tower: Unit) -> list[Robot]:
    reach = tower.range_of_attack()
    station = turn.station()
    robots = [
        robot for robot in turn.robots
        if robot.health > 0 and distance(tower.pos, robot.pos) <= reach
    ]

    def key(robot: Robot) -> tuple:
        hostile = robot.target_team in ("", turn.my_side)
        dist_station = (
            distance(robot.pos, station.pos) if station else 0
        )
        return (
            not hostile,
            robot.dizzy,
            dist_station,
            -robot.score,
            distance(tower.pos, robot.pos),
        )

    robots.sort(key=key)
    return robots


def gatling_targets(turn: Turn, tower: Unit) -> list[Pos]:
    candidates = _candidates(turn, tower)
    if not candidates:
        return []
    max_targets = tower.max_targets()
    chosen: list[Robot] = []
    for robot in candidates:
        if len(chosen) >= max_targets:
            break
        if any(_same_ray(tower.pos, other.pos, robot.pos) for other in chosen):
            continue
        if not cone_ok(tower.pos, [r.pos for r in (*chosen, robot)]):
            continue
        chosen.append(robot)
    return [robot.pos for robot in chosen]


def _same_ray(origin: Pos, near: Pos, far: Pos) -> bool:
    if distance(origin, far) <= distance(origin, near):
        return False
    cells = line_cells(origin, far)
    return near in cells


def railgun_target(turn: Turn, tower: Unit) -> Pos | None:
    candidates = _candidates(turn, tower)
    if not candidates:
        return None
    reach = tower.range_of_attack()
    energy = _railgun_energy(tower)
    alive = {robot.pos: robot for robot in turn.robots if robot.health > 0}
    best: tuple[float, Pos] | None = None
    for robot in candidates:
        cells = line_cells(tower.pos, robot.pos)[: reach + 1]
        remaining = energy
        gain = 0.0
        kills = 0
        for cell in cells:
            target = alive.get(cell)
            if target is None:
                continue
            dealt = min(remaining, target.health)
            gain += dealt
            if dealt >= target.health:
                kills += 1
                gain += target.score * 10
            remaining -= dealt
            if remaining <= 0:
                break
        value = gain + kills
        if best is None or value > best[0]:
            best = (value, robot.pos)
    return best[1] if best else None


def _railgun_energy(tower: Unit) -> float:
    power = tower.attack_power or 10
    return float(power * max(tower.level, 1) * 10)


def rocket_targets(turn: Turn, tower: Unit) -> list[Pos]:
    if tower.cooldown > 0:
        return []
    reach = tower.range_of_attack()
    robots = [
        robot for robot in turn.robots
        if robot.health > 0 and distance(tower.pos, robot.pos) <= reach
    ]
    if not robots:
        return []
    gains = _landing_gains(turn, robots)
    if not gains:
        return []
    ordered = sorted(gains.items(), key=lambda item: -item[1])
    chosen: list[Pos] = []
    total = 0.0
    for pos, gain in ordered:
        if len(chosen) >= tower.max_targets():
            break
        if any(distance(pos, other) < 2 for other in chosen):
            continue
        chosen.append(pos)
        total += gain
    station = turn.station()
    emergency = station is not None and any(
        distance(robot.pos, station.pos) <= 3 for robot in robots
    )
    if total < _ROCKET_MIN_GAIN and not emergency:
        return []
    return chosen


def _landing_gains(turn: Turn, robots: list[Robot]) -> dict[Pos, float]:
    spots: set[Pos] = set()
    for robot in robots:
        spots.add(robot.pos)
        for other in robots:
            if other is robot:
                continue
            if distance(robot.pos, other.pos) <= 2:
                mid = Pos(
                    (robot.pos.x + other.pos.x) // 2,
                    (robot.pos.y + other.pos.y) // 2,
                )
                spots.add(mid)
    gains: dict[Pos, float] = {}
    for spot in spots:
        gain = 0.0
        for robot in robots:
            d = distance(spot, robot.pos)
            if d == 0:
                dmg = _ROCKET_CENTER_DMG
            elif d == 1:
                dmg = _ROCKET_SPLASH_DMG
            else:
                continue
            gain += min(dmg, robot.health)
            if dmg >= robot.health:
                gain += robot.score * 10
        gains[spot] = gain
    return gains


def line_cells(start: Pos, end: Pos) -> list[Pos]:
    cells = [start]
    x0, y0 = start.x, start.y
    x1, y1 = end.x, end.y
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while (x0, y0) != (x1, y1):
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        cells.append(Pos(x0, y0))
    return cells


def _consumables(
    turn: Turn,
    pairs: tuple[tuple[Unit, Unit], ...],
    commands: dict[int, dict],
) -> None:
    walls = [wall for wall in turn.walls() if wall.health < wall_max(wall) * _WALL_FIX_RATIO]
    robots = [robot for robot in turn.robots if not robot.dizzy and robot.health > 0]
    for role, _tower in pairs:
        if role.unit_id in commands:
            continue
        if role.has("WallFixer") and walls:
            for wall in walls:
                if distance(role.pos, wall.pos) <= 1:
                    commands[role.unit_id] = use_command("WallFixer", wall.pos)
                    break
            if role.unit_id in commands:
                continue
        cluster = _cluster_center(robots)
        if cluster is not None:
            if role.has("Bomb"):
                commands[role.unit_id] = use_command("Bomb", cluster)
            elif role.has("DizzyWeapon"):
                commands[role.unit_id] = use_command("DizzyWeapon", cluster)


def wall_max(wall: Unit) -> int:
    return 1000 * max(wall.level, 1)


def _cluster_center(robots: list[Robot]) -> Pos | None:
    best: tuple[int, Pos] | None = None
    for robot in robots:
        if robot.kind not in ("largeRobot", "bossRobot"):
            continue
        group = [
            other for other in robots
            if distance(robot.pos, other.pos) <= 1
        ]
        if len(group) < 2:
            continue
        if best is None or len(group) > best[0]:
            best = (len(group), robot.pos)
    return best[1] if best else None
