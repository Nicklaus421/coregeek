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

# 建围墙前单次采石目标：攒够这一批再回基地建墙，避免一石一返的低效往返。
STONE_BATCH = 10


class Agent:
    def __init__(self) -> None:
        self._plan: BuildPlan | None = None
        self._plan_key: tuple | None = None
        self.task_engine = TaskEngine()
        # worker_id -> (ore, mine_pos_key)：持久化采矿目标，避免远距离路径摇摆
        self._mine_targets: dict[int, tuple[str, tuple[int, int]]] = {}
        # worker_id -> 本批采石目标数：攒够一批再回建墙，建完清零后重新采
        self._stone_goal: dict[int, int] = {}
        # 本回合已规划的移动目的地：避免两个角色同回合抢占同一格导致互相卡死
        self._reserved: set[tuple[int, int]] = set()

    # ---- 入口 ----
    def decide(self, payload: dict) -> dict:
        game = GameState(payload)
        self._ensure_plan(game)
        self._reserved = set()

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
        # 一名操作手负责操控三塔，白天临近夜晚时只需它回枢纽位，其余工人继续采矿
        operator = self._operator(game)
        op_id = operator.id if operator else None
        for w in game.workers:
            retreat = w.id == op_id and self._should_retreat(game, w)
            cmd = self._day_worker(game, w, claimed, retreat)
            if cmd is not None:
                cmds[w.id] = cmd
        return cmds

    def _day_worker(
        self, game: GameState, w: Role, claimed: set, retreat: bool
    ) -> dict | None:
        # 血量低且有药 -> 回血
        heal = self._heal(game, w)
        if heal is not None:
            return heal

        # 临近夜晚 -> 操作手撤回枢纽位待命
        if retreat:
            hub = self._plan.hub if self._plan else None
            if hub is not None and geo.cheb(w.pos, hub) > 0:
                return self._move_to(game, w, hub)
            return None

        # 1. 建满 3 座塔
        site = self._unbuilt_tower_site(game, claimed)
        if site is not None:
            # 先占坑再移动，避免两个工人同时抢同一个塔位
            claimed.add(site.key())
            if geo.cheb(w.pos, site) == 1:
                return {"action": "build", "name": ROCKET, "targetPos": [site.to_dict()]}
            return self._move_adjacent(game, w, site)

        # 2. 建围墙（批量采石后回建，从来袭方向开始）
        wall = self._unbuilt_wall(game, claimed)
        if wall is not None:
            stones = self._item_count(w, "stone")
            if stones == 0:
                # 没石头 -> 定下本批采石目标，出门采满一批再回
                self._stone_goal[w.id] = min(STONE_BATCH, self._remaining_walls(game))
                return self._go_collect(game, w, "stone")
            goal = self._stone_goal.get(w.id, 0)
            if goal > 0 and stones < goal:
                # 还没攒够一批，继续采
                return self._go_collect(game, w, "stone")
            # 攒够一批或已进入建墙阶段：把背包石头逐一建成墙，直到采完再返
            self._stone_goal[w.id] = 0
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
        if not towers:
            return cmds

        operator = self._operator(game)
        hub = self._plan.hub if self._plan else None

        # 操控手先占枢纽位；就位后每回合挑一座冷却完毕的塔发射
        if operator is not None:
            if hub is not None and geo.cheb(operator.pos, hub) > 0:
                mv = self._move_to(game, operator, hub)
                if mv is not None:
                    cmds[operator.id] = mv
            else:
                # 优先攻击威胁我方基地的机器人，无则攻击所有机器人刷分
                threats = [r for r in game.robots if r.target_team == game.team_type]
                target_pool = threats if threats else game.robots
                for tower in towers:
                    if tower.cooldown <= 0 and geo.cheb(operator.pos, tower.pos) <= 1:
                        target = choose_rocket_target(target_pool)
                        if target is not None:
                            cmds[tower.id] = {
                                "action": "attack",
                                "controllerId": str(operator.id),
                                "targetPos": [target.to_dict()],
                            }
                        break

        # 其余工人继续采集物资
        for w in game.workers:
            if operator is not None and w.id == operator.id:
                continue
            cmd = self._collect_sell(game, w, set())
            if cmd is not None:
                cmds[w.id] = cmd
        return cmds

    # ---- 通用 ----
    def _move_adjacent(self, game: GameState, role: Role, target: Pos) -> dict | None:
        blocked = game.blocked_set(exclude_role_id=role.id)
        # 目标格也视为障碍（避免走到要建/要采的格子上）；角色已站在目标格时除外，
        # 此时要把它挪到旁边一格去。
        if role.pos.key() != target.key():
            blocked.add(target.key())
        # 其他角色本回合已规划的移动目的地也视为障碍，避免同回合抢占同一格互相卡死
        blocked.update(self._reserved)
        best: list[Pos] | None = None
        for n in geo.neighbors(target, game.width, game.height):
            if n.key() in blocked:
                continue
            path = geo.bfs_path(role.pos, n, blocked, game.width, game.height)
            if path and (best is None or len(path) < len(best)):
                best = path
        if best:
            self._reserved.add(best[0].key())
            return {"action": "move", "targetPos": [best[0].to_dict()]}
        return None

    def _move_to(self, game: GameState, role: Role, target: Pos) -> dict | None:
        """移动到目标格本身（如枢纽位），目标格视为可达终点。"""
        blocked = game.blocked_set(exclude_role_id=role.id)
        blocked.update(self._reserved)
        path = geo.bfs_path(role.pos, target, blocked, game.width, game.height)
        if path:
            self._reserved.add(path[0].key())
            return {"action": "move", "targetPos": [path[0].to_dict()]}
        return None

    def _operator(self, game: GameState) -> Role | None:
        """选操控手：离枢纽格最近、背包较轻的工人。"""
        hub = self._plan.hub if self._plan else None
        if hub is None or not game.workers:
            return None
        return min(game.workers, key=lambda w: (geo.cheb(w.pos, hub), len(w.backpack)))

    def _heal(self, game: GameState, role: Role) -> dict | None:
        item = self._find_item(role, MEDICINE)
        if item is None:
            return None
        max_hp = MAX_HP.get(role.role_type, 999)
        if role.health < max_hp * 0.5:
            return {"action": "use", "name": item}
        return None

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

    def _remaining_walls(self, game: GameState) -> int:
        if self._plan is None:
            return 0
        built = {w.pos.key() for w in game.walls}
        return sum(1 for c in self._plan.wall_order if c.key() not in built)

    def _should_retreat(self, game: GameState, w: Role) -> bool:
        """操控手按到枢纽格距离动态决定是否回撤，其余工人不撤。"""
        hub = self._plan.hub if self._plan else None
        if hub is None:
            return False
        return game.rounds_until_night <= geo.cheb(w.pos, hub) + 2

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
        for ore in ("copper", "iron", "stone"):
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
