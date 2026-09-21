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


def attack_corner(base_cells: list[Pos], w: int, h: int) -> Pos:
    """推断机器人来袭角（地图四个角落之一）。

    基地固定在左上角或右下角：左上角基地被右上角来的机器人攻击，右下角基地被左下角攻击。
    即来袭角 = 与基地同处一条水平带（顶/底）、但 x 镜像到对侧的角落。
    """
    cx = sum(p.x for p in base_cells) / len(base_cells)
    cy = sum(p.y for p in base_cells) / len(base_cells)
    ax = w - 1 if cx < w / 2 else 0
    ay = h - 1 if cy > h / 2 else 0
    return Pos(ax, ay)


class BuildPlan:
    """围绕基地的固定建造蓝图：距离 1 圈放塔，距离 2 圈放围墙。

    围墙从来袭方向（attack_corner）优先建，背敌一侧留出入口供角色进出。
    """

    def __init__(self, base_cells: list[Pos], w: int, h: int):
        self.base_cells = base_cells
        self.attack = attack_corner(base_cells, w, h)
        self.ring1 = geo.ring_cells(base_cells, 1, w, h)
        self.ring2 = geo.ring_cells(base_cells, 2, w, h)
        self.tower_sites = self._pick_towers()
        self.wall_order = self._wall_order()

    def _pick_towers(self) -> list[Pos]:
        # 塔靠近来袭面摆放（火箭射程全图，摆位用于阻挡机器人）
        ring = list(self.ring1)
        ring.sort(key=lambda p: geo.cheb(p, self.attack))
        return ring[: len(TOWER_LOADOUT)]

    def _wall_order(self) -> list[Pos]:
        ring = list(self.ring2)
        if not ring:
            return []
        # 背敌一侧（离来袭角最远的 ring2 格）留作出入口
        entrance = max(ring, key=lambda p: geo.cheb(p, self.attack))
        order = [p for p in ring if p.key() != entrance.key()]
        # 从近敌一侧开始围，先把受攻击面封住
        order.sort(key=lambda p: geo.cheb(p, self.attack))
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
