import time
from typing import Any

from . import combat, economy, llm, news, tasks, treasure, validate
from .protocol import (
    Pos,
    Response,
    Turn,
    Unit,
    buy_command,
    distance,
    move_command,
    station_footprint,
)
from .state import blacklisted, maybe_reset, record_results, remember_cmds

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
            resp.commands = combat.night_commands(turn, state)
    except Exception:
        pass
    _filter(turn, state, resp)
    remember_cmds(state, resp.commands)
    return resp.dump()


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
    sites = _tower_sites(turn)
    order = _wall_order(turn)
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
        resp.commands[pioneer.unit_id] = tasks.submit_action(turn, state)
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
    _standby(turn, pioneer, resp)


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
        if not session.llm_plan_requested and session.text:
            resp.prompt = tasks.llm_plan_prompt(state)
            llm.note_sent(turn, state)
        elif (
            len(session.transcript) >= 4
            and not session.llm_extract_requested
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
        if not blacklisted(unit_id, cmd, state, turn.round_no)
    }


def _safe(func, *args):
    try:
        return func(*args)
    except Exception:
        return None


def _tower_sites(turn: Turn) -> tuple[Pos, ...]:
    station = turn.station()
    if station is None:
        return ()
    footprint = station_footprint(station.pos)
    cells = [
        pos for pos in _cells_at_distance(station.pos, 1) if turn.land(pos)
    ]
    cells.sort(key=lambda pos: (_footprint_distance(pos, footprint), pos.x, pos.y))
    return tuple(cells[:3])


def _wall_order(turn: Turn) -> tuple[Pos, ...]:
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
    entrance = Pos(xmax + 2, ymin - 1)
    return tuple(
        pos for pos in order if pos != entrance and turn.land(pos)
    )


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
