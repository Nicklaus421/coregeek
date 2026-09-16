from . import news
from .grid import next_step, step_adjacent
from .protocol import (
    GATLING,
    ORE_TYPES,
    RAILGUN,
    ROCKET,
    TOWER_TYPES,
    Pos,
    Turn,
    Unit,
    WALL,
    WALL_MATERIAL,
    WEAPON_BUILD_COST,
    build_command,
    buy_command,
    collect_command,
    distance,
    move_command,
    sell_command,
    use_command,
)
from .state import GameState

STONE_BATCH = 6
STONE_RESERVE = 12
SELL_BACKPACK_THRESHOLD = 60
SUMMON_ORDER_RESERVE = 150
TOWER_LOADOUT = (GATLING, RAILGUN, ROCKET)
_VOUCHER_TARGET = {
    "WeaponUpgradeVoucher1": TOWER_TYPES,
    "WeaponUpgradeVoucher2": TOWER_TYPES,
    "StationUpgradeVoucher1": ("station",),
    "StationUpgradeVoucher2": ("station",),
}


def shopping_list(turn: Turn, state: GameState) -> list[str]:
    plan: list[str] = []
    weapons = turn.weapons()
    levels = {unit.kind: unit.level for unit in weapons}
    shop = turn.shop_items
    gatling_lv = levels.get(GATLING, 1)
    railgun_lv = levels.get(RAILGUN, 1)
    rocket_lv = levels.get(ROCKET, 1)
    if gatling_lv == 1:
        plan.append("WeaponUpgradeVoucher1")
    if railgun_lv == 1:
        plan.append("WeaponUpgradeVoucher1")
    if gatling_lv == 2:
        plan.append("WeaponUpgradeVoucher2")
    plan.extend(["Medicine", "WallFixer", "WallFixer"])
    if rocket_lv == 1:
        plan.append("WeaponUpgradeVoucher1")
    if turn.gold > 400:
        plan.extend(["Bomb", "DizzyWeapon"])
    if turn.gold > 600 and state.summon_orders_today < 10:
        plan.append("SmallRobotSummonOrder")
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
    if towers_missing and turn.gold >= WEAPON_BUILD_COST:
        for index, site in enumerate(sites):
            if site in towers_missing and site not in claimed:
                _build_or_walk(
                    turn, role, site, TOWER_LOADOUT[index], claimed, commands,
                )
                return
    if builder:
        _builder_day(turn, role, walls_missing, claimed, commands)
    else:
        _trader_day(turn, state, role, claimed, commands)


def _builder_day(
    turn: Turn,
    role: Unit,
    walls_missing: list[Pos],
    claimed: set[Pos],
    commands: dict[int, dict],
) -> None:
    stones = role.count(WALL_MATERIAL)
    if walls_missing and stones:
        for site in walls_missing:
            if site not in claimed:
                _build_or_walk(turn, role, site, WALL, claimed, commands)
                return
    _mine(turn, role, WALL_MATERIAL, STONE_BATCH, claimed, commands)


def _trader_day(
    turn: Turn,
    state: GameState,
    role: Unit,
    claimed: set[Pos],
    commands: dict[int, dict],
) -> None:
    order = _summon_order_in_bag(role)
    if order is not None and state.summon_orders_today < 10:
        commands[role.unit_id] = use_command(order)
        state.summon_orders_today += 1
        return
    voucher = _usable_voucher(turn, role)
    if voucher is not None:
        target = _voucher_target(turn, voucher)
        if target is not None:
            if distance(role.pos, target) <= 1:
                commands[role.unit_id] = use_command(voucher, target)
                return
            if _walk(turn, role, target, claimed, commands):
                return

    sellable = _sellable(turn, state, role)
    if sellable:
        vendor = turn.vendor()
        if vendor is not None:
            if distance(role.pos, vendor) <= 1:
                name, num = sellable[0]
                commands[role.unit_id] = sell_command(name, num)
                return
            if _walk(turn, role, vendor, claimed, commands):
                return

    next_item = _next_purchase(turn, state, role)
    if next_item is not None:
        shop = turn.weapon_shop()
        price = turn.shop_items.get(next_item, 0)
        if shop is not None and turn.gold >= price and not role.backpack_full:
            if distance(role.pos, shop) <= 1:
                commands[role.unit_id] = buy_command(next_item, 1)
                return
            if _walk(turn, role, shop, claimed, commands):
                return

    ore = _ore_choice(turn, state)
    _mine(turn, role, ore, SELL_BACKPACK_THRESHOLD, claimed, commands)


def _ore_choice(turn: Turn, state: GameState) -> str:
    best_ore, best_value = WALL_MATERIAL, 0.0
    for ore in ORE_TYPES:
        mines = turn.mines(ore)
        if not mines:
            continue
        value = news.forecast(ore, state)
        if value > best_value:
            best_ore, best_value = ore, value
    return best_ore


def _sellable(
    turn: Turn,
    state: GameState,
    role: Unit,
) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    used = len(role.backpack)
    for ore in ORE_TYPES:
        count = role.count(ore)
        if not count:
            continue
        if ore == WALL_MATERIAL and count <= STONE_RESERVE and used < SELL_BACKPACK_THRESHOLD:
            continue
        sell_count = count if ore != WALL_MATERIAL else count - STONE_RESERVE
        if sell_count <= 0:
            continue
        if news.spiking(ore, state) or used >= SELL_BACKPACK_THRESHOLD:
            result.append((ore, sell_count))
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


def _usable_voucher(turn: Turn, role: Unit) -> str | None:
    for item in role.backpack:
        if item in _VOUCHER_TARGET and _voucher_target(turn, item) is not None:
            return item
    return None


def _voucher_target(turn: Turn, voucher: str) -> Pos | None:
    kinds = _VOUCHER_TARGET.get(voucher, ())
    want_level = 1 if voucher.endswith("1") else 2
    for unit in turn.ours:
        if unit.kind in kinds and unit.level == want_level and unit.health > 0:
            return unit.pos
    return None


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
    role: Unit,
    ore: str,
    batch: int,
    claimed: set[Pos],
    commands: dict[int, dict],
) -> bool:
    if role.backpack_full:
        return False
    mines = sorted(
        turn.mines(ore),
        key=lambda pos: (distance(role.pos, pos), pos.x, pos.y),
    )
    for mine in mines:
        if role.pos != mine and distance(role.pos, mine) <= 1:
            if role.count(ore) < batch or ore == WALL_MATERIAL:
                commands[role.unit_id] = collect_command(mine)
                claimed.add(mine)
                return True
            return False
        if _walk(turn, role, mine, claimed, commands):
            return True
    return False
