from math import atan2, degrees
from typing import Any

from .protocol import (
    ACTIONS,
    CONTROLLABLE_TYPES,
    GATLING,
    PIONEER,
    Pos,
    RAILGUN,
    TOWER_TYPES,
    Turn,
    Unit,
    WALL,
    WORKER,
    distance,
)

_NEEDS_TARGET_ONE = {"move", "remove", "collect"}
_NEEDS_NAME = {"sell", "buy", "drop"}
_VOUCHERS = {
    "WeaponUpgradeVoucher1", "WeaponUpgradeVoucher2",
    "WallUpgradeVoucher1", "WallUpgradeVoucher2",
    "StationUpgradeVoucher1", "StationUpgradeVoucher2",
}
_TARGETED_USE = _VOUCHERS | {"WallFixer", "DizzyWeapon", "Bomb"}


def cone_ok(origin: Pos, targets: list[Pos]) -> bool:
    if len(targets) <= 1:
        return True
    angles = [
        degrees(atan2(t.y - origin.y, t.x - origin.x))
        for t in targets
        if t != origin
    ]
    if len(angles) <= 1:
        return True
    for i, a in enumerate(angles):
        for b in angles[i + 1:]:
            diff = abs(a - b) % 360
            if min(diff, 360 - diff) > 90:
                return False
    return True


def check(turn: Turn, unit: Unit | None, cmd: dict[str, Any]) -> bool:
    action = str(cmd.get("action") or "")
    if action not in ACTIONS:
        return False
    if unit is None or unit.health <= 0:
        return False

    targets = [
        Pos.load(raw) for raw in cmd.get("targetPos") or ()
        if isinstance(raw, dict) and "x" in raw and "y" in raw
    ]
    raw_targets = cmd.get("targetPos") or []
    if len(targets) != len(raw_targets):
        return False
    for pos in targets:
        if not 0 <= pos.x < turn.width or not 0 <= pos.y < turn.height:
            return False

    name = cmd.get("name")
    if action in _NEEDS_NAME and not name:
        return False
    if action in _NEEDS_TARGET_ONE and len(targets) != 1:
        return False

    if action == "move":
        return distance(unit.pos, targets[0]) == 1 and turn.land(targets[0])
    if action == "collect":
        return (
            unit.kind == WORKER
            and distance(unit.pos, targets[0]) <= 1
            and turn.zones.get(targets[0]) in ("stone", "iron", "copper")
        )
    if action == "build":
        return (
            unit.kind == WORKER
            and turn.is_day
            and name in (*TOWER_TYPES, WALL)
            and len(targets) == 1
            and distance(unit.pos, targets[0]) <= 1
            and turn.land(targets[0])
        )
    if action == "remove":
        return unit.kind == WORKER and distance(unit.pos, targets[0]) <= 1
    if action == "attack":
        if turn.is_day or unit.kind not in TOWER_TYPES:
            return False
        controller = _find(turn, cmd.get("controllerId"))
        if controller is None or controller.kind not in CONTROLLABLE_TYPES:
            return False
        if distance(controller.pos, unit.pos) > 1 or unit.cooldown > 0:
            return False
        if not targets or len(targets) > unit.max_targets():
            return False
        reach = unit.range_of_attack()
        if any(distance(unit.pos, pos) > reach for pos in targets):
            return False
        if unit.kind == RAILGUN and len(targets) != 1:
            return False
        if unit.kind == GATLING and not cone_ok(unit.pos, targets):
            return False
        return True
    if action == "sell":
        return (
            name in ("stone", "iron", "copper")
            and unit.count(str(name)) >= int(cmd.get("num") or 1)
            and _near_zone(turn, unit, "vendor")
        )
    if action == "buy":
        return _near_zone(turn, unit, "weaponShop")
    if action == "use":
        if not unit.has(str(name)):
            return False
        if name in _TARGETED_USE:
            if len(targets) != 1:
                return False
            if name in _VOUCHERS or name == "WallFixer":
                return distance(unit.pos, targets[0]) <= 1
        return True
    if action == "drop":
        return unit.has(str(name))
    if action == "acceptTask":
        return unit.kind == PIONEER and _near_task_point(turn, unit)
    if action == "submitAnswer":
        return unit.kind == PIONEER and bool(cmd.get("taskAnswer"))
    if action == "summonTreasure":
        items = cmd.get("item") or []
        return (
            unit.kind == PIONEER
            and len(targets) == 1
            and distance(unit.pos, targets[0]) <= 1
            and items
            and all(unit.has(str(item)) for item in items)
        )
    return True


def filter_all(
    turn: Turn,
    commands: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for unit_id, cmd in commands.items():
        unit = _find(turn, unit_id)
        try:
            if check(turn, unit, cmd):
                result[unit_id] = cmd
        except Exception:
            continue
    return result


def _find(turn: Turn, raw_id: Any) -> Unit | None:
    try:
        unit_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    for unit in turn.ours:
        if unit.unit_id == unit_id:
            return unit
    return None


def _near_zone(turn: Turn, unit: Unit, kind: str) -> bool:
    return any(
        distance(unit.pos, pos) <= 1
        for pos, zone in turn.zones.items()
        if zone == kind
    )


def _near_task_point(turn: Turn, unit: Unit) -> bool:
    prefix = "challenger" if turn.my_side == "challenger" else "defender"
    return any(
        distance(unit.pos, pos) <= 1
        for pos, zone in turn.zones.items()
        if zone.startswith(prefix)
    )
