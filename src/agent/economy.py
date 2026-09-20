"""经济与建造规划：常量、建造蓝图、火箭目标选择。"""
from __future__ import annotations

from . import geometry as geo
from .models import Pos, Robot
from .state import ROCKET

# 三座塔全用火箭发射台（射程全图、无弹道阻挡、中心 20 + 溅射 10）。
TOWER_LOADOUT = (ROCKET, ROCKET, ROCKET)

# 商店商品名
WEAPON_UPGRADE_1 = "WeaponUpgradeVoucher1"
WEAPON_UPGRADE_2 = "WeaponUpgradeVoucher2"
STATION_UPGRADE_1 = "StationUpgradeVoucher1"
STATION_UPGRADE_2 = "StationUpgradeVoucher2"
WALL_UPGRADE_1 = "WallUpgradeVoucher1"
WALL_UPGRADE_2 = "WallUpgradeVoucher2"
WALL_FIXER = "WallFixer"
MEDICINE = "Medicine"
DIZZY_WEAPON = "DizzyWeapon"
BOMB = "Bomb"

# 矿石类型
ORE_TYPES = ("stone", "iron", "copper")

# 机器人积分 / 优先级权重
ROBOT_VALUE = {
    "bossRobot": 10,
    "largeRobot": 4,
    "middleRobot": 2,
    "smallRobot": 1,
}


class BuildPlan:
    """围绕基地的固定建造蓝图：距离 1 圈放塔，距离 2 圈放围墙（留背敌出入口）。"""

    def __init__(self, base_cells: list[Pos], enemy_station: Pos | None, w: int, h: int):
        self.base_cells = base_cells
        self.enemy = enemy_station
        self.ring1 = geo.ring_cells(base_cells, 1, w, h)
        self.ring2 = geo.ring_cells(base_cells, 2, w, h)
        self.tower_sites = self._pick_towers()
        self.wall_order = self._wall_order()

    def _pick_towers(self) -> list[Pos]:
        ring = list(self.ring1)
        if self.enemy is not None:
            ring.sort(key=lambda p: geo.cheb(p, self.enemy))
        return ring[: len(TOWER_LOADOUT)]

    def _wall_order(self) -> list[Pos]:
        ring = list(self.ring2)
        if not ring:
            return []
        if self.enemy is not None:
            # 背敌一侧（离敌方最远的 ring2 格）留作出入口
            entrance = max(ring, key=lambda p: geo.cheb(p, self.enemy))
        else:
            entrance = ring[0]
        order = [p for p in ring if p.key() != entrance.key()]
        if self.enemy is not None:
            # 从近敌一侧开始围，先把受攻击面封住
            order.sort(key=lambda p: geo.cheb(p, self.enemy))
        return order


def choose_rocket_target(robots: list[Robot]) -> Pos | None:
    """为火箭选择落点：优先覆盖高价值机器人集群（中心 20 + 周围 8 格溅射 10）。"""
    if not robots:
        return None

    # 候选落点 = 各机器人位置及其 8 邻域（去重）
    candidates: set[tuple[int, int]] = set()
    for r in robots:
        candidates.add(r.pos.key())
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.add((r.pos.x + dx, r.pos.y + dy))

    best_pos: tuple[int, int] | None = None
    best_val = -1
    for c in candidates:
        val = 0
        for r in robots:
            d = geo.cheb_xy(c[0], c[1], r.pos.x, r.pos.y)
            if d == 0:
                val += ROBOT_VALUE.get(r.role_type, 1) * 2  # 中心 20 点，权重更高
            elif d <= 1:
                val += ROBOT_VALUE.get(r.role_type, 1)  # 溅射 10 点
        if val > best_val:
            best_val = val
            best_pos = c

    if best_pos is None:
        return None
    return Pos(best_pos[0], best_pos[1])
