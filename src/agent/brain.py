"""Agent 决策核心。

- 白天：工人优先采石头建围墙（防御），再建塔、买/用升级券（武器优先）、采集贩卖；开拓者做自进化任务。
- 夜晚：角色操控火箭塔攻击机器人；开拓者若任务进行中则继续做任务。

每回合每个角色至多一条指令；attack 指令的 key 为武器 id，``controllerId`` 为操控角色 id。
"""
from __future__ import annotations

from . import geometry as geo
from .economy import (
    MEDICINE,
    ORE_TYPES,
    ROCKET,
    STATION_UPGRADE_1,
    STATION_UPGRADE_2,
    WALL_UPGRADE_1,
    WALL_UPGRADE_2,
    WEAPON_UPGRADE_1,
    WEAPON_UPGRADE_2,
    BuildPlan,
    choose_rocket_target,
)
from .models import Pos, Role, Zone
from .state import GameState
from .tasks import TaskEngine

MAX_HP = {"worker": 220, "pioneer": 200}


class Agent:
    def __init__(self) -> None:
        self._plan: BuildPlan | None = None
        self._plan_key: tuple | None = None
        self.task_engine = TaskEngine()
        # worker_id -> (ore, mine_pos_key)：持久化采矿目标，避免远距离路径摇摆
        self._mine_targets: dict[int, tuple[str, tuple[int, int]]] = {}

    # ---- 入口 ----
    def decide(self, payload: dict) -> dict:
        game = GameState(payload)
        self._ensure_plan(game)

        if game.is_night:
            cmds = self._night_plan(game)
        else:
            cmds = self._day_plan(game)

        # 自进化任务：白天主动接取/推进；夜晚仅在任务进行中继续提交/执行沙盒
        if game.is_night and not game.phase_task:
            task_cmd, execute_cmd, prompt = None, None, ""
        else:
            task_cmd, execute_cmd, prompt = self.task_engine.step(game)

        role_command_map: dict[str, dict] = {str(k): v for k, v in cmds.items()}
        if task_cmd is not None and game.pioneer is not None:
            role_command_map[str(game.pioneer.id)] = task_cmd

        return {
            "roleCommandMap": role_command_map,
            "prompt": prompt or "",
            "executeCmd": execute_cmd or "",
        }

    # ---- 建造蓝图缓存 ----
    def _ensure_plan(self, game: GameState) -> None:
        base = game.base_cells()
        if not base:
            self._plan = None
            self._plan_key = None
            return
        key = tuple(sorted(c.key() for c in base))
        if self._plan_key != key:
            self._plan = BuildPlan(base, game.width, game.height)
            self._plan_key = key

    # ---- 白天 ----
    def _day_plan(self, game: GameState) -> dict:
        cmds: dict[int, dict] = {}
        claimed: set[tuple[int, int]] = set()
        retreat = game.rounds_until_night <= 12
        # 撤离时按稳定分配去各自的塔，避免多人挤同一塔
        tower_for: dict[int, Role] = {}
        if retreat:
            for w, tower in self._assign_workers_to_towers(game):
                tower_for[w.id] = tower
        for w in game.workers:
            cmd = self._day_worker(game, w, claimed, retreat, tower_for.get(w.id))
            if cmd is not None:
                cmds[w.id] = cmd
        return cmds

    def _day_worker(
        self, game: GameState, w: Role, claimed: set, retreat: bool, tower: Role | None
    ) -> dict | None:
        # 血量低且有药 -> 回血
        heal = self._heal(game, w)
        if heal is not None:
            return heal

        # 临近夜晚 -> 撤回各自分配的塔旁待命
        if retreat:
            if tower is not None and geo.cheb(w.pos, tower.pos) > 1:
                return self._move_adjacent(game, w, tower.pos)
            return None

        # 1. 建满 3 座塔
        site = self._unbuilt_tower_site(game, claimed)
        if site is not None:
            # 先占坑再移动，避免两个工人同时抢同一个塔位
            claimed.add(site.key())
            if geo.cheb(w.pos, site) == 1:
                return {"action": "build", "name": ROCKET, "targetPos": [site.to_dict()]}
            return self._move_adjacent(game, w, site)

        # 2. 建围墙（需采石头，从来袭方向开始）
        wall = self._unbuilt_wall(game, claimed)
        if wall is not None:
            if not self._has_item(w, "stone"):
                return self._go_collect(game, w, "stone")
            # 有石头才占坑，避免无石工人空占位置
            claimed.add(wall.key())
            if geo.cheb(w.pos, wall) == 1:
                return {"action": "build", "name": "wall", "targetPos": [wall.to_dict()]}
            return self._move_adjacent(game, w, wall)

        # 3. 升级流程（武器优先）
        cmd = self._upgrade_flow(game, w, claimed)
        if cmd is not None:
            return cmd

        # 4. 采集 + 贩卖攒金币（石头优先）
        cmd = self._collect_sell(game, w, claimed)
        if cmd is not None:
            return cmd

        return None

    # ---- 夜晚 ----
    def _night_plan(self, game: GameState) -> dict:
        cmds: dict[int, dict] = {}
        towers = game.towers

        # 工人 → 塔稳定分配（与白天撤离一致）
        pairs = self._assign_workers_to_towers(game)
        used = {t.id for _, t in pairs}
        # 开拓者没在任务中时，补位剩余塔
        if game.pioneer is not None and not game.phase_task:
            remaining = [t for t in towers if t.id not in used]
            if remaining:
                pairs.append((game.pioneer, remaining[0]))

        # 优先攻击威胁我方基地的机器人，无则攻击所有机器人刷分
        threats = [r for r in game.robots if r.target_team == game.team_type]
        target_pool = threats if threats else game.robots

        for op, tower in pairs:
            if geo.cheb(op.pos, tower.pos) <= 1:
                if tower.cooldown <= 0:
                    target = choose_rocket_target(target_pool)
                    if target is not None:
                        cmds[tower.id] = {
                            "action": "attack",
                            "controllerId": str(op.id),
                            "targetPos": [target.to_dict()],
                        }
                continue
            mv = self._move_adjacent(game, op, tower.pos)
            if mv is not None:
                cmds[op.id] = mv
        return cmds

    # ---- 通用 ----
    def _move_adjacent(self, game: GameState, role: Role, target: Pos) -> dict | None:
        blocked = game.blocked_set(exclude_role_id=role.id)
        # 目标格也视为障碍（避免走到要建/要采的格子上）；角色已站在目标格时除外，
        # 此时要把它挪到旁边一格去。
        if role.pos.key() != target.key():
            blocked.add(target.key())
        best: list[Pos] | None = None
        for n in geo.neighbors(target, game.width, game.height):
            if n.key() in blocked:
                continue
            path = geo.bfs_path(role.pos, n, blocked, game.width, game.height)
            if path and (best is None or len(path) < len(best)):
                best = path
        if best:
            return {"action": "move", "targetPos": [best[0].to_dict()]}
        return None

    def _heal(self, game: GameState, role: Role) -> dict | None:
        item = self._find_item(role, MEDICINE)
        if item is None:
            return None
        max_hp = MAX_HP.get(role.role_type, 999)
        if role.health < max_hp * 0.5:
            return {"action": "use", "name": item}
        return None

    def _assign_workers_to_towers(self, game: GameState) -> list[tuple[Role, Role]]:
        """把工人贪心分配到塔（每塔就近选未分配的工人），白天撤离与夜晚操控共用，保证稳定不摇摆。"""
        towers = list(game.towers)
        available = list(game.workers)
        pairs: list[tuple[Role, Role]] = []
        for tower in towers:
            if not available:
                break
            w = min(available, key=lambda o: geo.cheb(o.pos, tower.pos))
            available.remove(w)
            pairs.append((w, tower))
        return pairs

    def _unbuilt_tower_site(self, game: GameState, claimed: set) -> Pos | None:
        if self._plan is None:
            return None
        built = {t.pos.key() for t in game.towers}
        for site in self._plan.tower_sites:
            if site.key() not in built and site.key() not in claimed:
                return site
        return None

    def _unbuilt_wall(self, game: GameState, claimed: set) -> Pos | None:
        if self._plan is None:
            return None
        built = {w.pos.key() for w in game.walls}
        for cell in self._plan.wall_order:
            if cell.key() not in built and cell.key() not in claimed:
                return cell
        return None

    def _go_collect(self, game: GameState, w: Role, ore: str) -> dict | None:
        mine = self._select_mine(game, w, ore)
        if mine is None:
            return None
        if geo.cheb(w.pos, mine.pos) <= 1:
            return {"action": "collect", "targetPos": [mine.pos.to_dict()]}
        return self._move_adjacent(game, w, mine.pos)

    def _mine_at(self, game: GameState, ore: str, key: tuple[int, int]) -> Zone | None:
        for z in game.mines:
            if z.neutral_type == ore and z.pos.key() == key:
                return z
        return None

    def _select_mine(self, game: GameState, w: Role, ore: str) -> Zone | None:
        """选矿：远距离时锁定同一矿（持久目标）避免来回摇摆；到矿旁后自然就近采。

        矿有多个资源格，允许多个工人采同一矿，故不按矿整体占坑，避免矿少时工人无矿可采而呆立。
        """
        t = self._mine_targets.get(w.id)
        if t is not None and t[0] == ore:
            mine = self._mine_at(game, ore, t[1])
            if mine is not None and geo.cheb(w.pos, mine.pos) > 1:
                return mine
        mine = self._nearest_mine(game, w, ore)
        if mine is not None:
            self._mine_targets[w.id] = (ore, mine.pos.key())
        return mine

    def _nearest_mine(self, game: GameState, w: Role, ore: str) -> Zone | None:
        mines = [z for z in game.mines if z.neutral_type == ore]
        if not mines:
            return None
        return min(mines, key=lambda z: geo.cheb(w.pos, z.pos))

    def _collect_sell(self, game: GameState, w: Role, claimed: set) -> dict | None:
        ore = self._ore_to_sell(game, w)
        if ore is not None:
            vendor = game.vendor
            if vendor is None:
                return None
            if geo.cheb(w.pos, vendor.pos) <= 1:
                return {"action": "sell", "name": ore, "num": self._item_count(w, ore)}
            return self._move_adjacent(game, w, vendor.pos)
        for ore in ("stone", "copper", "iron"):
            if self._nearest_mine(game, w, ore) is not None:
                return self._go_collect(game, w, ore)
        return None

    def _ore_to_sell(self, game: GameState, w: Role) -> str | None:
        prices = game.vendor_prices()
        best: str | None = None
        best_price = -1
        for ore in ORE_TYPES:
            if self._has_item(w, ore):
                p = prices.get(ore, 0)
                if p > best_price:
                    best = ore
                    best_price = p
        return best

    def _upgrade_flow(self, game: GameState, w: Role, claimed: set) -> dict | None:
        up = self._next_upgrade(game)
        if up is None:
            return None
        voucher, target, price = up
        if price is None:
            return None
        item = self._find_item(w, voucher)
        if item is not None:
            if geo.cheb(w.pos, target.pos) <= 1:
                return {"action": "use", "name": item, "targetPos": [target.pos.to_dict()]}
            return self._move_adjacent(game, w, target.pos)
        if game.gold >= price:
            shop = game.weapon_shop
            if shop is None:
                return None
            if geo.cheb(w.pos, shop.pos) <= 1:
                return {"action": "buy", "name": voucher, "num": 1}
            return self._move_adjacent(game, w, shop.pos)
        return None

    def _next_upgrade(self, game: GameState):
        # 武器优先（按等级从低到高，先全部到 2 级再上 3 级）
        for tower in sorted(game.towers, key=lambda t: t.level):
            if tower.level < 3:
                voucher = WEAPON_UPGRADE_1 if tower.level == 1 else WEAPON_UPGRADE_2
                return voucher, tower, game.shop_price(voucher)
        st = game.station
        if st is not None and st.level < 3:
            voucher = STATION_UPGRADE_1 if st.level == 1 else STATION_UPGRADE_2
            return voucher, st, game.shop_price(voucher)
        for wall in game.walls:
            if wall.level < 3:
                voucher = WALL_UPGRADE_1 if wall.level == 1 else WALL_UPGRADE_2
                return voucher, wall, game.shop_price(voucher)
        return None

    # ---- 背包工具 ----
    @staticmethod
    def _find_item(role: Role, name: str) -> str | None:
        for it in role.backpack:
            if it.lower() == name.lower():
                return it
        return None

    @staticmethod
    def _has_item(role: Role, name: str) -> bool:
        return Agent._find_item(role, name) is not None

    @staticmethod
    def _item_count(role: Role, name: str) -> int:
        return sum(1 for it in role.backpack if it.lower() == name.lower())
