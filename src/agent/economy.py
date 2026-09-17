from . import news
from .grid import next_step, step_adjacent
from .protocol import (
    ORE_TYPES,
    PIONEER,
    ROCKET,
    STATION,
    TOWER_TYPES,
    Pos,
    Turn,
    Unit,
    WALL,
    WALL_MATERIAL,
    WEAPON_BUILD_COST,
    WORKER,
    build_command,
    buy_command,
    collect_command,
    distance,
    move_command,
    sell_command,
    use_command,
)
from .state import GameState, attack_direction, facing_score

STONE_BATCH = 6  # 一次采矿攒够砌多段墙，避免"挖一块跑一趟"
STONE_RESERVE = 2
STONE_STOCK_CAP = 12
BUILD_NEARBY = 2
SELL_BACKPACK_THRESHOLD = 40
SELL_ORE_THRESHOLD = 12
TRIP_SLACK = 20  # 背包剩这么多个位置就去跑一趟卖货
TRIP_AFTER_ROUND = 45  # 下午准时跑一趟：卖货 + 采购，顺路在天黑前回基地
TRIP_MIN_ORE = 12
DUSK_ROUND = 55
SUMMON_ORDER_RESERVE = 150
BOMB_THRESHOLD = 250
STATION_HEAL_RATIO = 0.75  # 基地血量低于该比例就买升级券（升级即满血）
# 三座塔全用火箭发射台：射程覆盖全图、单发 20 点还带溅射，性价比最高
TOWER_LOADOUT = (ROCKET, ROCKET, ROCKET)
STATION_BASE_HP = 1500
_MAX_HEALTH = {WORKER: 220, PIONEER: 200}
# 一处矿点全图共享 10 次采集后消失；按计划采集次数摊薄路上的时间
_MINE_CAPACITY = 10
_MINE_PLAN = 8
_VOUCHER_TARGET = {
    "WeaponUpgradeVoucher1": TOWER_TYPES,
    "WeaponUpgradeVoucher2": TOWER_TYPES,
    "WallUpgradeVoucher1": (WALL,),
    "WallUpgradeVoucher2": (WALL,),
    "StationUpgradeVoucher1": (STATION,),
    "StationUpgradeVoucher2": (STATION,),
}


def shopping_list(turn: Turn, state: GameState) -> list[str]:
    """金币优先级：武器升级 > 基地抢修 > 修墙/自愈 > 围墙升级 > 反击道具。

    生存分（score3）权重最高，所以钱先花在“少掉血”和“多击杀”上：
    塔升级同时提高射程与多目标数；基地升级券直接把基地奶满。
    召唤令只是把机器人塞给对手，不产生收益，不买。
    """
    plan: list[str] = []
    towers = turn.weapons()
    shop = turn.shop_items
    station = turn.station()
    # 1. 三塔升级：先全体到 2 级，再冲 3 级（缺的塔还没造出来，买了券也用不了）。
    #    按"当前处于该等级的塔数"买券，同型多塔各算一份，不能去重。
    for want_level, voucher in (
        (1, "WeaponUpgradeVoucher1"),
        (2, "WeaponUpgradeVoucher2"),
    ):
        plan.extend(
            voucher for tower in towers if tower.level == want_level
        )
    # 2. 基地被打残：升级券 = 满血 + 提上限，直接换生存分
    if station is not None and station.health <= STATION_BASE_HP * STATION_HEAL_RATIO:
        plan.extend(["StationUpgradeVoucher1", "Medicine"])
    # 3. 消耗品：修墙比自愈更值（墙替基地挡刀）
    plan.extend(["WallFixer", "WallFixer", "Medicine", "Medicine"])
    # 4. 围墙升级（20/30 金一级，血量翻倍，性价比最高）
    if turn.gold > 120:
        plan.extend(["WallUpgradeVoucher1", "WallUpgradeVoucher1"])
    # 5. 盈余够多才买一次性的范围清场道具
    if turn.gold > BOMB_THRESHOLD:
        plan.extend(["Bomb", "DizzyWeapon"])
    return [item for item in plan if item in shop]


def worker_day(
    turn: Turn,
    state: GameState,
    role: Unit,
    builder: bool,
    sites: tuple[Pos, ...],
    towers_missing: list[Pos],
    walls_missing: list[Pos],
    claimed: set[Pos],
    commands: dict[int, dict],
) -> None:
    if _use_medicine(role, commands):
        return
    if towers_missing and turn.gold >= WEAPON_BUILD_COST:
        for index, site in enumerate(sites):
            if site in towers_missing and site not in claimed:
                _build_or_walk(
                    turn, role, site, TOWER_LOADOUT[index], claimed, commands,
                )
                return
    _prune_mines(turn, state)
    # 傍晚回防：天黑前必须回到基地附近操控武器
    station = turn.station()
    if (
        turn.day_round >= DUSK_ROUND
        and station is not None
        and distance(role.pos, station.pos) > 3
    ):
        if builder and walls_missing and role.count(WALL_MATERIAL):
            _build_or_walk(turn, role, walls_missing[0], WALL, claimed, commands)
            if role.unit_id in commands:
                return
        _walk(turn, role, station.pos, claimed, commands)
        return
    if builder:
        _builder_day(turn, state, role, walls_missing, claimed, commands)
    else:
        _trader_day(turn, state, role, claimed, commands)


def _use_medicine(role: Unit, commands: dict[int, dict]) -> bool:
    """半血就吃药，别等角色被打死（少一个单位就少一个操控手）。"""
    cap = _MAX_HEALTH.get(role.kind)
    if cap is None or not role.has("Medicine"):
        return False
    if role.health * 2 > cap:
        return False
    commands[role.unit_id] = use_command("Medicine")
    return True


def _prune_mines(turn: Turn, state: GameState) -> None:
    """矿点会消失/刷新，定期清掉过期的认领记录，避免字典无限增长。"""
    if len(state.mine_used) <= 64 and len(state.mine_target) <= 16:
        return
    alive = set()
    for ore in ORE_TYPES:
        alive.update(turn.mines(ore))
    state.mine_used = {
        mine: used for mine, used in state.mine_used.items() if mine in alive
    }
    state.mine_target = {
        uid: mine for uid, mine in state.mine_target.items() if mine in alive
    }


def _builder_day(
    turn: Turn,
    state: GameState,
    role: Unit,
    walls_missing: list[Pos],
    claimed: set[Pos],
    commands: dict[int, dict],
) -> None:
    stones = role.count(WALL_MATERIAL)
    mine = _adjacent_mine(turn, role)
    # 紧缺位置的墙就在手边就立刻砌，否则攒够一批再走，避免一块石头跑一趟
    urgent = bool(walls_missing) and (
        mine is None or distance(role.pos, walls_missing[0]) <= BUILD_NEARBY
    )
    if walls_missing and stones and (urgent or stones >= STONE_BATCH):
        for site in walls_missing:
            if site not in claimed:
                _build_or_walk(turn, role, site, WALL, claimed, commands)
                return
    # 石料够修完剩下的墙之后就不再挖石，转去挖更值钱的矿
    if stones < _stone_need(walls_missing):
        _mine(turn, state, role, WALL_MATERIAL, STONE_BATCH, claimed, commands)
        return
    if not walls_missing:
        # 墙圈已完工：别再囤石，改走"采矿+卖货+采购"，否则包满了会呆站
        _trader_day(turn, state, role, claimed, commands)
        return
    _mine(
        turn, state, role, _ore_choice(turn, state, role),
        SELL_BACKPACK_THRESHOLD, claimed, commands,
    )


def _stone_need(walls_missing: list[Pos]) -> int:
    return max(STONE_RESERVE, min(len(walls_missing), STONE_STOCK_CAP))


def _adjacent_mine(turn: Turn, role: Unit) -> Pos | None:
    for mine in turn.mines(WALL_MATERIAL):
        if role.pos != mine and distance(role.pos, mine) <= 1:
            return mine
    return None


def _trader_day(
    turn: Turn,
    state: GameState,
    role: Unit,
    claimed: set[Pos],
    commands: dict[int, dict],
) -> None:
    """默认挖矿，只有"快满包/价格飙升"才跑一趟商店（卖货顺路采购）。

    采集是每回合 1 个，走路不产出，所以采购行程越少越好：
    把一趟分解成 卖货 -> 买券 -> 用券 -> 回矿点，期间不做无谓折返。
    """
    order = _summon_order_in_bag(role)
    if order is not None and state.summon_orders_today < 10:
        commands[role.unit_id] = use_command(order)
        state.summon_orders_today += 1
        return
    voucher = _usable_voucher(turn, state, role)
    vendor = turn.vendor()
    running = role.unit_id in state.shop_run
    sellable = _sellable(turn, state, role)
    if sellable and vendor is not None and (running or _trip_needed(turn, state, role)):
        state.shop_run.add(role.unit_id)
        if distance(role.pos, vendor) <= 1:
            name, num = sellable[0]
            commands[role.unit_id] = sell_command(name, num)
            return
        if _walk(turn, role, vendor, claimed, commands):
            return
    if running:
        if _shop_errand(turn, state, role, claimed, commands):
            return
        state.shop_run.discard(role.unit_id)
    if voucher is not None:
        target = _voucher_target(turn, state, voucher)
        if target is not None:
            if distance(role.pos, target) <= 1:
                commands[role.unit_id] = use_command(voucher, target)
                return
            if _walk(turn, role, target, claimed, commands):
                return
    ore = _ore_choice(turn, state, role)
    _mine(
        turn, state, role, ore, role.capacity or 100, claimed, commands,
    )


def _trip_needed(turn: Turn, state: GameState, role: Unit) -> bool:
    """值得为卖货专门跑一趟：包快满了、下午了（顺路在天黑前回基地）、或者正在涨价。"""
    if sum(role.count(ore) for ore in ORE_TYPES) < TRIP_MIN_ORE:
        return False
    capacity = role.capacity or 100
    if len(role.backpack) >= capacity - TRIP_SLACK:
        return True
    if turn.day_round >= TRIP_AFTER_ROUND:
        return True
    return any(
        news.spiking(ore, state) for ore in ORE_TYPES if role.count(ore) > 0
    )


def _shop_errand(
    turn: Turn,
    state: GameState,
    role: Unit,
    claimed: set[Pos],
    commands: dict[int, dict],
) -> bool:
    """采购行程：买下当前买得起且有目标可用的一件，买不动了行程结束。"""
    item = _next_purchase(turn, state, role)
    shop = turn.weapon_shop()
    if item is None or shop is None:
        return False
    if turn.gold < turn.shop_items.get(item, 0) or role.backpack_full:
        return False
    if distance(role.pos, shop) <= 1:
        commands[role.unit_id] = buy_command(item, 1)
        return True
    return _walk(turn, role, shop, claimed, commands)


def _ore_choice(turn: Turn, state: GameState, role: Unit) -> str:
    """按“单位时间收益”选矿：收益高但太远的矿，摊上路上时间后往往不如近矿。"""
    best_ore, best_value = WALL_MATERIAL, -1.0
    for ore in ORE_TYPES:
        mine = _pick_mine(turn, state, role, ore)
        if mine is None:
            continue
        value = _amortized(state, ore, role.pos, mine)
        if value > best_value:
            best_ore, best_value = ore, value
    return best_ore


def _remaining(state: GameState, mine: Pos) -> int:
    """估算矿点残留采集次数（全图共享 10 次），未知时保守按满算。"""
    return max(1, _MINE_CAPACITY - state.mine_used.get(mine, 0))


def _amortized(state: GameState, ore: str, origin: Pos, mine: Pos) -> float:
    planned = min(_MINE_PLAN, _remaining(state, mine))
    travel = distance(origin, mine)
    return news.forecast(ore, state) * planned / (travel + planned)


def _mine_candidates(
    turn: Turn, state: GameState, role: Unit, ore: str,
) -> list[Pos]:
    """候选矿点：先续挖上次认领的矿，其余按“单位时间收益”排序。

    返回列表而不是单点，是为了在落脚点被同伴占住时还能退而求其次，
    否则该回合会直接空转。
    """
    mines = turn.mines(ore)
    if not mines:
        return []
    ordered = sorted(
        mines,
        key=lambda pos: (
            -_amortized(state, ore, role.pos, pos),
            distance(role.pos, pos),
            pos.x,
            pos.y,
        ),
    )
    committed = state.mine_target.get(role.unit_id)
    if committed in mines and _remaining(state, committed) > 1:
        ordered.remove(committed)
        ordered.insert(0, committed)
    return ordered


def _pick_mine(turn: Turn, state: GameState, role: Unit, ore: str) -> Pos | None:
    candidates = _mine_candidates(turn, state, role, ore)
    return candidates[0] if candidates else None


def _sellable(
    turn: Turn,
    state: GameState,
    role: Unit,
) -> list[tuple[str, int]]:
    """可卖清单（按预测价降序）。石料留够修墙储备，其余全卖。

    是否值得为它跑一趟由 `_trip_needed` 决定，这里只管"有什么能卖"。
    """
    result: list[tuple[str, int]] = []
    for ore in ORE_TYPES:
        count = role.count(ore)
        if ore == WALL_MATERIAL:
            count = max(0, count - STONE_RESERVE)
        if count > 0:
            result.append((ore, count))
    result.sort(key=lambda item: -news.forecast(item[0], state))
    return result


def _next_purchase(turn: Turn, state: GameState, role: Unit) -> str | None:
    for item in shopping_list(turn, state):
        if role.has(item):
            continue
        price = turn.shop_items.get(item)
        if price is not None and turn.gold >= price:
            return item
    return None


_SUMMON_ORDERS = (
    "SmallRobotSummonOrder",
    "MiddleRobotSummonOrder",
    "LargeRobotSummonOrder",
    "BossRobotSummonOrder",
)


def _summon_order_in_bag(role: Unit) -> str | None:
    for name in _SUMMON_ORDERS:
        if role.has(name):
            return name
    return None


def _usable_voucher(turn: Turn, state: GameState, role: Unit) -> str | None:
    for item in role.backpack:
        if item in _VOUCHER_TARGET and _voucher_target(turn, state, item) is not None:
            return item
    return None


def _voucher_target(turn: Turn, state: GameState, voucher: str) -> Pos | None:
    """优先升级迎敌方向的工事；墙则先修最破的那一段。"""
    kinds = _VOUCHER_TARGET.get(voucher, ())
    if not kinds:
        return None
    want_level = 1 if voucher.endswith("1") else 2
    candidates = [
        unit for unit in turn.ours
        if unit.kind in kinds and unit.level == want_level and unit.health > 0
    ]
    if not candidates:
        return None
    station = turn.station()
    if station is None:
        return candidates[0].pos
    bearing = attack_direction(turn, state)
    candidates.sort(
        key=lambda unit: (
            -facing_score(unit.pos, station.pos, bearing),
            unit.health,
        )
    )
    return candidates[0].pos


def _build_or_walk(
    turn: Turn,
    role: Unit,
    target: Pos,
    name: str,
    claimed: set[Pos],
    commands: dict[int, dict],
) -> None:
    if role.pos != target and distance(role.pos, target) <= 1:
        commands[role.unit_id] = build_command(target, name)
        claimed.add(target)
        return
    _walk(turn, role, target, claimed, commands)


def _walk(
    turn: Turn,
    role: Unit,
    target: Pos,
    claimed: set[Pos],
    commands: dict[int, dict],
) -> bool:
    step = step_adjacent(turn, role, target)
    if step is None or step in claimed:
        return False
    claimed.add(step)
    commands[role.unit_id] = move_command(step)
    return True


def _mine(
    turn: Turn,
    state: GameState,
    role: Unit,
    ore: str,
    batch: int,
    claimed: set[Pos],
    commands: dict[int, dict],
) -> bool:
    """采一次矿。站在矿边就继续采（零路程），否则走向认领的矿点。

    采集是每回合 1 个，路途才是主要成本，所以一旦认领就一路挖到矿点消失/换矿，
    不做“每次重新挑最近的矿”这种会来回折返的决策。
    """
    if role.backpack_full:
        return False
    if ore != WALL_MATERIAL and role.count(ore) >= batch:
        return False
    mine = _adjacent_ore(turn, role, ore)
    if mine is not None:
        state.mine_target[role.unit_id] = mine
        state.mine_used[mine] = state.mine_used.get(mine, 0) + 1
        commands[role.unit_id] = collect_command(mine)
        claimed.add(mine)
        return True
    for mine in _mine_candidates(turn, state, role, ore):
        if _walk(turn, role, mine, claimed, commands):
            state.mine_target[role.unit_id] = mine
            return True
    return False


def _adjacent_ore(turn: Turn, role: Unit, ore: str) -> Pos | None:
    for mine in turn.mines(ore):
        if role.pos != mine and distance(role.pos, mine) <= 1:
            return mine
    return None
