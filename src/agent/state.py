"""游戏状态解析与派生信息。"""
from __future__ import annotations

from .models import PlayerTask, Pos, Robot, Role, ShopItem, Zone

# roleType 取值
PIONEER = "pioneer"
WORKER = "worker"
STATION = "station"
WALL = "wall"
GATLING = "gatling"
RAILGUN = "railgun"
ROCKET = "rocket"

WEAPON_TYPES = (GATLING, RAILGUN, ROCKET)

# neutralType 取值
MINE_TYPES = ("stone", "iron", "copper")
VENDOR = "vendor"
WEAPON_SHOP = "weaponShop"

ROUNDS_PER_DAY = 130
DAY_ROUNDS = 70


class GameState:
    def __init__(self, payload: dict):
        self.raw = payload
        self.round_no = int(payload.get("roundNo", 0))

        map_info = payload.get("mapInfo") or {}
        self.width = int(map_info.get("width", 41))
        self.height = int(map_info.get("height", 32))
        self.zones: list[Zone] = [Zone.from_dict(z) for z in (map_info.get("zones") or [])]

        team = payload.get("teamOur") or {}
        self.team_type = team.get("type", "challenger")
        self.team_id = str(team.get("teamId", ""))
        self.team_name = team.get("teamName", "")
        self.gold = int(team.get("goldNum", 0))
        self.total_score = int(team.get("totalScore", 0))
        self.player_tasks: list[PlayerTask] = [PlayerTask.from_dict(t) for t in (team.get("playerTasks") or [])]
        self.roles: list[Role] = [Role.from_dict(r) for r in (team.get("roles") or [])]

        enemy = payload.get("teamEnemy") or {}
        self.enemy_roles: list[Role] = [Role.from_dict(r) for r in (enemy.get("roles") or [])]

        robot = payload.get("robot") or {}
        self.robots: list[Robot] = [Robot.from_dict(r) for r in (robot.get("roles") or [])]

        self.phase_task = payload.get("phaseTask", "") or ""
        self.last_summon_treasure_result = int(payload.get("lastSummonTreasureResult", 0))
        self.llm_resp = payload.get("llmResp", "") or ""
        world_news = payload.get("worldNews") or {}
        self.official_news = world_news.get("officialNews", "") or ""
        self.folk_legends = world_news.get("folkLegends", "") or ""
        self.last_cmd_result = payload.get("lastCmdResult", "") or ""
        self.vendor_shop_list: list[ShopItem] = [ShopItem.from_dict(i) for i in (payload.get("vendorShopList") or [])]
        self.weapon_shop_list: list[ShopItem] = [ShopItem.from_dict(i) for i in (payload.get("weaponShopList") or [])]
        self.errors: list[dict] = payload.get("errors") or []
        self.last_round_role_action_results: dict = payload.get("lastRoundRoleActionResults") or {}

        # 索引
        self.roles_by_id: dict[int, Role] = {r.id: r for r in self.roles}
        self.enemy_by_id: dict[int, Role] = {r.id: r for r in self.enemy_roles}
        self.robots_by_id: dict[int, Robot] = {r.id: r for r in self.robots}

    # ---- 时间 ----
    @property
    def day(self) -> int:
        return self.round_no // ROUNDS_PER_DAY + 1

    @property
    def round_in_day(self) -> int:
        return self.round_no % ROUNDS_PER_DAY

    @property
    def is_night(self) -> bool:
        return self.round_in_day >= DAY_ROUNDS

    @property
    def rounds_until_night(self) -> int:
        return max(0, DAY_ROUNDS - self.round_in_day)

    # ---- 单位 ----
    def roles_of_type(self, t: str) -> list[Role]:
        return [r for r in self.roles if r.role_type == t]

    @property
    def pioneer(self) -> Role | None:
        for r in self.roles:
            if r.role_type == PIONEER:
                return r
        return None

    @property
    def workers(self) -> list[Role]:
        return self.roles_of_type(WORKER)

    @property
    def station(self) -> Role | None:
        for r in self.roles:
            if r.role_type == STATION:
                return r
        return None

    @property
    def towers(self) -> list[Role]:
        return [r for r in self.roles if r.role_type in WEAPON_TYPES]

    @property
    def walls(self) -> list[Role]:
        return self.roles_of_type(WALL)

    @property
    def enemy_station(self) -> Role | None:
        for r in self.enemy_roles:
            if r.role_type == STATION:
                return r
        return None

    # ---- 中立元素 ----
    def zones_of_type(self, t: str) -> list[Zone]:
        return [z for z in self.zones if z.neutral_type == t]

    @property
    def mines(self) -> list[Zone]:
        return [z for z in self.zones if z.neutral_type in MINE_TYPES]

    @property
    def vendor(self) -> Zone | None:
        zs = self.zones_of_type(VENDOR)
        return zs[0] if zs else None

    @property
    def weapon_shop(self) -> Zone | None:
        zs = self.zones_of_type(WEAPON_SHOP)
        return zs[0] if zs else None

    def own_task_points(self) -> list[Zone]:
        prefix = "challenger" if self.team_type == "challenger" else "defender"
        return [z for z in self.zones if z.neutral_type.startswith(prefix + "TaskPoint")]

    # ---- 基地 ----
    def base_cells(self) -> list[Pos]:
        """基地 2x2，station 的 pos 为左上角坐标。"""
        st = self.station
        if st is None:
            return []
        x, y = st.pos.x, st.pos.y
        # 左上角 -> 占据 (x,y),(x+1,y),(x,y-1),(x+1,y-1)
        return [Pos(x, y), Pos(x + 1, y), Pos(x, y - 1), Pos(x + 1, y - 1)]

    # ---- 占用（用于寻路，所有会阻挡移动的元素） ----
    def blocked_set(self, exclude_role_id: int | None = None) -> set[tuple[int, int]]:
        blocked: set[tuple[int, int]] = set()
        for r in self.roles:
            if exclude_role_id is not None and r.id == exclude_role_id:
                continue
            blocked.add(r.pos.key())
            # 基地占 4 格
            if r.role_type == STATION:
                blocked.update(c.key() for c in self.base_cells())
        for r in self.enemy_roles:
            blocked.add(r.pos.key())
            if r.role_type == STATION:
                for c in self._enemy_base_cells(r):
                    blocked.add(c.key())
        for rb in self.robots:
            blocked.add(rb.pos.key())
        for z in self.zones:
            blocked.add(z.pos.key())
        return blocked

    @staticmethod
    def _enemy_base_cells(st: Role) -> list[Pos]:
        x, y = st.pos.x, st.pos.y
        return [Pos(x, y), Pos(x + 1, y), Pos(x, y - 1), Pos(x + 1, y - 1)]

    def vendor_prices(self) -> dict[str, int]:
        return {i.name: i.price for i in self.vendor_shop_list}

    def shop_price(self, name: str) -> int | None:
        for i in self.weapon_shop_list:
            if i.name == name:
                return i.price
        return None
