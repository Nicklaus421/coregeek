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
    三塔聚成一簇，围绕一个「枢纽格」摆放，让一个工人站在枢纽格即可同时操控三塔。
    """

    def __init__(self, base_cells: list[Pos], w: int, h: int):
        self.base_cells = base_cells
        self.w = w
        self.h = h
        self.attack = attack_corner(base_cells, w, h)
        self.ring1 = geo.ring_cells(base_cells, 1, w, h)
        self.ring2 = geo.ring_cells(base_cells, 2, w, h)
        self.hub = Pos(0, 0)
        self.tower_sites = self._pick_towers()
        self.wall_order = self._wall_order()

    def _pick_towers(self) -> list[Pos]:
        # station 为基地左上角（最小 x、最大 y）；据此把三塔聚成一簇。
        st_x = min(p.x for p in self.base_cells)
        st_y = max(p.y for p in self.base_cells)
        cx = sum(p.x for p in self.base_cells) / len(self.base_cells)
        if cx < self.w / 2:
            # 基地靠左（左上角）：塔建在基地左下角，枢纽格 (st_x-1, st_y-1)
            self.hub = Pos(st_x - 1, st_y - 1)
            return [Pos(st_x, st_y - 2), Pos(st_x - 1, st_y - 2), Pos(st_x - 1, st_y)]
        # 基地靠右（右下角）：塔建在基地右下角，枢纽格 (st_x+2, st_y-1)
        self.hub = Pos(st_x + 2, st_y - 1)
        return [Pos(st_x + 2, st_y), Pos(st_x + 1, st_y - 2), Pos(st_x + 2, st_y - 2)]

    def _wall_order(self) -> list[Pos]:
        ring = list(self.ring2)
        if not ring:
            return []
        # 背敌一侧（离来袭角最远的 ring2 格）留作出入口
        entrance = max(ring, key=lambda p: geo.cheb(p, self.attack))
        order = [p for p in ring if p.key() != entrance.key()]
        # 枢纽格旁留通道：操作手白天要出去采集、夜晚要回枢纽操控，不能被墙圈死
        order = [p for p in order if geo.cheb(p, self.hub) > 1]
        # 从近敌一侧开始围，先把受攻击面封住
        order.sort(key=lambda p: geo.cheb(p, self.attack))
        return order


def choose_rocket_targets(robots: list[Robot], n: int) -> list[Pos]:
    """为火箭塔选 n 个落点（n = 塔等级，决定每轮发射的导弹数）。

    贪心逐枚分配：每枚优先覆盖尚未被命中机器人的高价值集群（中心 20 权重 x2、
    溅射 10 权重 x1）；当所有机器人至少被覆盖一次后，剩余导弹叠在价值最高的集群上
    （落点重叠伤害可叠加）。
    """
    if not robots or n <= 0:
        return []

    # 候选落点 = 各机器人位置及其 8 邻域（去重）
    candidates: set[tuple[int, int]] = set()
    for r in robots:
        candidates.add(r.pos.key())
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.add((r.pos.x + dx, r.pos.y + dy))

    targets: list[Pos] = []
    hit: set[int] = set()  # 已被前序导弹覆盖的机器人下标
    for _ in range(n):
        best: tuple[int, int] | None = None
        best_marginal = -1
        best_raw = -1
        for c in candidates:
            marginal = 0  # 尚未被命中机器人的价值
            raw = 0       # 全部机器人价值（用于全部命中后叠加）
            for i, r in enumerate(robots):
                d = geo.cheb_xy(c[0], c[1], r.pos.x, r.pos.y)
                w = ROBOT_VALUE.get(r.role_type, 1)
                if d == 0:
                    raw += w * 2
                    if i not in hit:
                        marginal += w * 2
                elif d <= 1:
                    raw += w
                    if i not in hit:
                        marginal += w
            # 优先覆盖未命中机器人；全部命中后按原始价值叠加到高价值集群
            if marginal > best_marginal or (marginal == best_marginal and raw > best_raw):
                best_marginal = marginal
                best_raw = raw
                best = c
        if best is None:
            break
        targets.append(Pos(best[0], best[1]))
        for i, r in enumerate(robots):
            if geo.cheb_xy(best[0], best[1], r.pos.x, r.pos.y) <= 1:
                hit.add(i)
    return targets
