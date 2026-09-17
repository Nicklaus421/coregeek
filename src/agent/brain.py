import time
from typing import Any

from . import combat, economy, llm, news, tasks, trace, treasure, validate
from .protocol import (
    Pos,
    Response,
    Turn,
    Unit,
    buy_command,
    distance,
    move_command,
    station_footprint,
    use_command,
)
from .state import (
    attack_direction,
    blacklisted,
    facing_score,
    maybe_reset,
    record_attack,
    record_results,
    remember_cmds,
)

TIME_BUDGET = 3.5
_NEIGHBOUR_STEPS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    turn = Turn.load(payload)
    state = maybe_reset(turn)
    record_results(turn, state)
    resp = Response()
    try:
        _ingest(turn, state)
        if turn.is_day:
            _day(turn, state, resp)
        else:
            record_attack(turn, state)
            resp.commands = combat.night_commands(turn, state)
    except Exception:
        pass
    _filter(turn, state, resp)
    remember_cmds(state, resp.commands)
    dumped = resp.dump()
    trace.record(payload, dumped, (time.monotonic() - started) * 1000)
    return dumped


def _ingest(turn: Turn, state) -> None:
    for code, _desc in turn.errors:
        if code == 5:
            llm.note_quota_error(state)
    _safe(news.ingest, turn, state)
    _safe(treasure.ingest_legend, turn, state)
    _safe(treasure.handle_result, turn, state)
    _safe(tasks.handle_answer_error, turn, state)
    _apply_llm_resp(turn, state)


def _apply_llm_resp(turn: Turn, state) -> None:
    text = turn.llm_resp.strip()
    if not text:
        return
    session = state.task
    if session.llm_plan_requested and not session.llm_plan_applied:
        if tasks.apply_llm_plan(state, text):
            session.llm_plan_applied = True
        return
    if session.llm_extract_requested and not session.llm_extract_applied:
        if tasks.apply_llm_extract(state, text):
            session.llm_extract_applied = True
        return
    if state.treasure.llm_requested_day:
        treasure.apply_llm_answer(state, text)


def _day(turn: Turn, state, resp: Response) -> None:
    sites = _tower_sites(turn, state)
    order = _wall_order(turn, state)
    standing_towers = {unit.pos for unit in turn.weapons()}
    standing_walls = {unit.pos for unit in turn.walls()}
    occupied = turn.occupied_cells()
    if len(standing_towers) >= 3:
        towers_missing = []
    else:
        towers_missing = [
            pos for pos in sites
            if pos not in standing_towers
        ][: 3 - len(standing_towers)]
    walls_missing = [pos for pos in order if pos not in standing_walls]
    free_towers = [pos for pos in towers_missing if pos not in occupied]
    free_walls = [pos for pos in walls_missing if pos not in occupied]

    claimed: set[Pos] = set()
    pioneer = turn.pioneer()
    if pioneer is not None:
        _pioneer_day(turn, state, pioneer, resp)

    workers = turn.workers()
    for index, role in enumerate(workers):
        _safe(
            economy.worker_day,
            turn, state, role, index == 0, sites,
            list(free_towers), list(free_walls), claimed, resp.commands,
        )

    resp.execute_cmd = tasks.attach_exec(turn, state)
    _attach_prompt(turn, state, resp)


def _pioneer_day(turn: Turn, state, pioneer: Unit, resp: Response) -> None:
    command = _safe(tasks.pioneer_action, turn, state)
    if command is not None:
        resp.commands[pioneer.unit_id] = command
        return
    if state.task.active and _safe(tasks.answer_ready, turn, state):
        command = _safe(tasks.submit_action, turn, state)
        if command is not None:
            resp.commands[pioneer.unit_id] = command
        return
    if state.task.active:
        return
    command = _safe(treasure.pioneer_action, turn, state)
    if command is not None:
        resp.commands[pioneer.unit_id] = command
        return
    command = _buy_task_items(turn, state, pioneer)
    if command is not None:
        resp.commands[pioneer.unit_id] = command
        return
    command = _pioneer_upkeep(turn, pioneer)
    if command is not None:
        resp.commands[pioneer.unit_id] = command
        return
    _standby(turn, pioneer, resp)


_PIONEER_MAX_HP = 200


def _pioneer_upkeep(turn: Turn, pioneer: Unit) -> dict | None:
    """开拓者阵亡会直接丢掉任务分，半血就自救，顺路在商店备一份药。"""
    if pioneer.health * 2 > _PIONEER_MAX_HP:
        return None
    if pioneer.has("Medicine"):
        return use_command("Medicine")
    shop = turn.weapon_shop()
    price = turn.shop_items.get("Medicine")
    if shop is None or price is None or turn.gold < price:
        return None
    if distance(pioneer.pos, shop) <= 1:
        return buy_command("Medicine", 1)
    step = step_adjacent(turn, pioneer, shop)
    return move_command(step) if step is not None else None


def _buy_task_items(turn: Turn, state, pioneer: Unit) -> dict | None:
    from .grid import step_adjacent

    case = state.treasure
    if case.done or not case.item_candidates:
        return None
    missing = [
        item for item in case.item_candidates[0] if not pioneer.has(item)
    ]
    if not missing:
        return None
    shop = turn.weapon_shop()
    if shop is None or pioneer.backpack_full:
        return None
    item = missing[0]
    price = turn.shop_items.get(item)
    if price is None or turn.gold < price:
        return None
    if distance(pioneer.pos, shop) <= 1:
        return buy_command(item, 1)
    step = step_adjacent(turn, pioneer, shop)
    if step is not None:
        return move_command(step)
    return None


def _standby(turn: Turn, pioneer: Unit, resp: Response) -> None:
    from .grid import step_adjacent

    station = turn.station()
    if station is None:
        return
    if distance(pioneer.pos, station.pos) <= 4:
        return
    step = step_adjacent(turn, pioneer, station.pos)
    if step is not None:
        resp.commands[pioneer.unit_id] = move_command(step)


def _attach_prompt(turn: Turn, state, resp: Response) -> None:
    if not llm.budget_ok(turn, state):
        return
    session = state.task
    if session.active:
        if session.pending_cmds:
            return
        explored = len(session.transcript)
        if not session.draft_answer and explored >= session.llm_plan_at + 2:
            # 迭代修复：每执行两条命令就请一次 LLM 给出下一步修复动作
            resp.prompt = tasks.llm_plan_prompt(state)
            llm.note_sent(turn, state)
        elif (
            not session.draft_answer
            and not session.llm_extract_requested
            and explored >= 8
        ):
            resp.prompt = tasks.llm_extract_prompt(state)
            llm.note_sent(turn, state)
        return
    case = state.treasure
    if (
        not case.done
        and case.legends
        and (not case.loc_candidates or not case.item_candidates)
        and case.llm_requested_day != turn.day
    ):
        resp.prompt = treasure.legend_prompt(state)
        case.llm_requested_day = turn.day
        llm.note_sent(turn, state)


def _filter(turn: Turn, state, resp: Response) -> None:
    filtered = validate.filter_all(turn, resp.commands)
    resp.commands = {
        unit_id: cmd
        for unit_id, cmd in filtered.items()
        if cmd and not blacklisted(unit_id, cmd, state, turn.round_no)
    }


def _safe(func, *args):
    try:
        return func(*args)
    except Exception:
        return None


def _tower_sites(turn: Turn, state) -> tuple[Pos, ...]:
    station = turn.station()
    if station is None:
        return ()
    footprint = station_footprint(station.pos)
    cells = [
        pos for pos in _cells_at_distance(station.pos, 1) if turn.land(pos)
    ]
    if not cells:
        return ()
    bearing = _bearing(state, turn, station.pos)
    # 面向进攻方向选第一座塔，其余两座紧邻以便同一批工人操作
    anchor = min(
        cells,
        key=lambda pos: (
            -_bearing_score(pos, station.pos, bearing),
            _footprint_distance(pos, footprint),
            pos.x,
            pos.y,
        ),
    )
    rest = sorted(
        (pos for pos in cells if pos != anchor),
        key=lambda pos: (distance(pos, anchor), pos.x, pos.y),
    )
    return tuple([anchor, *rest[:2]])


def _wall_order(turn: Turn, state) -> tuple[Pos, ...]:
    station = turn.station()
    if station is None:
        return ()
    footprint = station_footprint(station.pos)
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    order = [
        *(Pos(x, ymin - 2) for x in range(xmax + 2, xmin - 3, -1)),
        *(Pos(xmin - 2, y) for y in range(ymin - 1, ymax + 2)),
        *(Pos(x, ymax + 2) for x in range(xmin - 2, xmax + 3)),
        *(Pos(xmax + 2, y) for y in range(ymax + 1, ymin - 2, -1)),
    ]
    ring = list(dict.fromkeys(pos for pos in order if turn.land(pos)))
    if not ring:
        return ()
    bearing = _bearing(state, turn, station.pos)
    entrance = _pick_entrance(state, station.pos, ring, bearing)
    walls = [pos for pos in ring if pos != entrance]
    if bearing is not None:
        walls.sort(
            key=lambda pos: (
                -_bearing_score(pos, station.pos, bearing),
                pos.x,
                pos.y,
            )
        )
    return tuple(walls)


_ENTRANCE_MARGIN = 0.34  # 新出口要比旧出口明显更背向敌人，避免来回改口


def _pick_entrance(
    state,
    station_pos: Pos,
    ring: list[Pos],
    bearing: tuple[float, float] | None,
) -> Pos:
    """出入口留在背向进攻方向的一侧，随观测到的敌情缓慢迁移。"""
    if bearing is None:
        return state.wall_entrance or ring[-1]
    candidate = min(
        ring,
        key=lambda pos: (_bearing_score(pos, station_pos, bearing), pos.x, pos.y),
    )
    current = state.wall_entrance
    if current not in ring or (
        _bearing_score(current, station_pos, bearing)
        > _bearing_score(candidate, station_pos, bearing) + _ENTRANCE_MARGIN
    ):
        state.wall_entrance = candidate
    return state.wall_entrance


def _bearing(state, turn: Turn, station_pos: Pos) -> tuple[float, float] | None:
    """进攻方向：优先夜间实测，其次敌方基地方向，最后“指向地图内部”的先验。

    机器人从远离我方最近边缘的一侧（地图内部 / 敌方基地那一角）压过来，
    而不是从最近的边缘来；所以无实测数据时先验要指向地图中心方向。
    """
    return attack_direction(turn, state)


def _bearing_score(
    pos: Pos, station_pos: Pos, bearing: tuple[float, float] | None,
) -> float:
    return facing_score(pos, station_pos, bearing)


def _cells_at_distance(station_pos: Pos, radius: int) -> tuple[Pos, ...]:
    footprint = station_footprint(station_pos)
    xs = [pos.x for pos in footprint]
    ys = [pos.y for pos in footprint]
    cells = []
    for x in range(min(xs) - radius, max(xs) + radius + 1):
        for y in range(min(ys) - radius, max(ys) + radius + 1):
            pos = Pos(x, y)
            if pos in footprint:
                continue
            if _footprint_distance(pos, footprint) == radius:
                cells.append(pos)
    return tuple(cells)


def _footprint_distance(pos: Pos, footprint: tuple[Pos, ...]) -> int:
    if not footprint:
        return 0
    return min(distance(pos, cell) for cell in footprint)


def _neighbours(pos: Pos) -> tuple[Pos, ...]:
    return tuple(
        Pos(pos.x + dx, pos.y + dy) for dx, dy in _NEIGHBOUR_STEPS
    )
