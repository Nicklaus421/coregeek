from dataclasses import dataclass, field
from typing import Any

DAY_ROUNDS = 70
NIGHT_ROUNDS = 60
ROUNDS_PER_DAY = DAY_ROUNDS + NIGHT_ROUNDS
MAX_ROUNDS = 1300

WEAPON_BUILD_COST = 25
WALL_MATERIAL = "stone"
ORE_TYPES = ("stone", "iron", "copper")
LAND = "land"
STATION = "station"
WALL = "wall"
WORKER = "worker"
PIONEER = "pioneer"
GATLING = "gatling"
RAILGUN = "railgun"
ROCKET = "rocket"
TOWER_TYPES = (GATLING, RAILGUN, ROCKET)
CONTROLLABLE_TYPES = (WORKER, PIONEER)
VENDOR = "vendor"
WEAPON_SHOP = "weaponShop"
TOWER_RANGE_BY_LEVEL = {
    GATLING: (3, 5, 7),
    RAILGUN: (6, 8, 10),
    ROCKET: (10, 15, 10**9),
}
ROBOT_SCORE = {
    "smallRobot": 1,
    "middleRobot": 2,
    "largeRobot": 4,
    "bossRobot": 10,
}
ROBOT_ATTACK = {
    "smallRobot": 5,
    "middleRobot": 10,
    "largeRobot": 20,
    "bossRobot": 40,
}
TASK_POINT_TYPES = (
    "challengerTaskPoint1",
    "challengerTaskPoint2",
    "defenderTaskPoint1",
    "defenderTaskPoint2",
)
CHALLENGER = "challenger"
DEFENDER = "defender"

ACTIONS = (
    "move", "attack", "sell", "buy", "build", "remove",
    "acceptTask", "submitAnswer", "summonTreasure", "use", "drop",
    "collect",
)


@dataclass(frozen=True, slots=True)
class Pos:
    x: int
    y: int

    @classmethod
    def load(cls, raw: Any) -> "Pos":
        return cls(int(raw["x"]), int(raw["y"]))

    def dump(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


def distance(first: Pos, second: Pos) -> int:
    return max(abs(first.x - second.x), abs(first.y - second.y))


def station_footprint(pos: Pos) -> tuple[Pos, ...]:
    return (
        pos,
        Pos(pos.x + 1, pos.y),
        Pos(pos.x, pos.y - 1),
        Pos(pos.x + 1, pos.y - 1),
    )


@dataclass(frozen=True, slots=True)
class Unit:
    unit_id: int
    pos: Pos
    kind: str
    health: int
    level: int
    cooldown: int
    attack_range: int
    attack_power: int
    capacity: int | None
    backpack: tuple[str, ...]

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Unit":
        raw_capacity = raw.get("backPackCapability")
        return cls(
            int(raw.get("id") or 0),
            Pos.load(raw["pos"]),
            str(raw["roleType"]),
            int(raw.get("health") or 0),
            int(raw.get("level") or 0),
            int(raw.get("cooldown") or 0),
            int(raw.get("attackRange") or 0),
            int(raw.get("attackPower") or 0),
            int(raw_capacity) if raw_capacity is not None else None,
            tuple(str(item) for item in raw.get("backpack") or ()),
        )

    @property
    def backpack_full(self) -> bool:
        if self.capacity is None:
            return False
        return len(self.backpack) >= self.capacity

    def count(self, name: str) -> int:
        return self.backpack.count(name)

    def has(self, name: str) -> bool:
        return name in self.backpack

    def range_of_attack(self) -> int:
        if self.attack_range > 0:
            return self.attack_range
        table = TOWER_RANGE_BY_LEVEL.get(self.kind)
        if table is None:
            return 0
        level = min(max(self.level, 1), len(table))
        return table[level - 1]

    def max_targets(self) -> int:
        return min(max(self.level, 1), 3)


@dataclass(frozen=True, slots=True)
class Robot:
    robot_id: int
    pos: Pos
    kind: str
    health: int
    dizzy: bool
    target_team: str

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Robot":
        return cls(
            int(raw.get("id") or 0),
            Pos.load(raw["pos"]),
            str(raw.get("roleType") or ""),
            int(raw.get("health") or 0),
            str(raw.get("abnormalState") or "") == "dizzy",
            str(raw.get("targetTeam") or ""),
        )

    @property
    def score(self) -> int:
        return ROBOT_SCORE.get(self.kind, 1)


@dataclass(frozen=True, slots=True)
class TaskPoint:
    task_type: str
    pos: Pos
    cooldown: int
    score_reward: int
    gold_reward: int
    is_valid: bool
    timeout_rounds: int

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "TaskPoint":
        return cls(
            str(raw.get("taskType") or ""),
            Pos.load(raw["taskPosition"]),
            int(raw.get("coldDownRounds") or 0),
            int(raw.get("scoreReward") or 0),
            int(raw.get("goldReward") or 0),
            bool(raw.get("isValid")),
            int(raw.get("timeoutRounds") or 0),
        )


def _price_map(raw: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for entry in raw or ():
        result[str(entry.get("name"))] = int(entry.get("price") or 0)
    return result


@dataclass(frozen=True, slots=True)
class Turn:
    round_no: int
    is_day: bool
    day: int
    day_round: int
    gold: int
    width: int
    height: int
    zones: dict[Pos, str]
    ours: tuple[Unit, ...]
    enemy: tuple[Unit, ...]
    robots: tuple[Robot, ...]
    my_side: str
    total_score: int
    task_points: tuple[TaskPoint, ...]
    vendor_prices: dict[str, int]
    shop_items: dict[str, int]
    official_news: str
    folk_legend: str
    phase_task: str
    last_cmd_result: str
    llm_resp: str
    last_summon: int
    last_results: dict[int, bool]
    errors: tuple[tuple[int, str], ...]

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "Turn":
        round_no = int(payload["roundNo"])
        info = payload["mapInfo"]
        team = payload["teamOur"]
        day = (round_no - 1) // ROUNDS_PER_DAY + 1
        news = payload.get("worldNews") or {}
        return cls(
            round_no,
            (round_no - 1) % ROUNDS_PER_DAY < DAY_ROUNDS,
            day,
            (round_no - 1) % ROUNDS_PER_DAY,
            int(team.get("goldNum") or 0),
            int(info["width"]),
            int(info["height"]),
            {
                Pos.load(zone["pos"]): str(zone["neutralType"])
                for zone in info.get("zones") or ()
            },
            tuple(Unit.load(role) for role in team.get("roles") or ()),
            tuple(
                Unit.load(role)
                for role in (payload.get("teamEnemy") or {}).get("roles") or ()
            ),
            tuple(
                Robot.load(robot)
                for robot in (payload.get("robot") or {}).get("roles") or ()
            ),
            str(team.get("type") or CHALLENGER),
            int(team.get("totalScore") or 0),
            tuple(
                TaskPoint.load(task) for task in team.get("playerTasks") or ()
            ),
            _price_map(payload.get("vendorShopList")),
            _price_map(payload.get("weaponShopList")),
            str(news.get("officialNews") or ""),
            str(news.get("folkLegends") or ""),
            str(payload.get("phaseTask") or ""),
            str(payload.get("lastCmdResult") or ""),
            str(payload.get("llmResp") or ""),
            int(payload.get("lastSummonTreasureResult") or 0),
            {
                int(key): bool(value)
                for key, value in (
                    payload.get("lastRoundRoleActionResults") or {}
                ).items()
            },
            tuple(
                (int(err.get("errorCode") or 0), str(err.get("description") or ""))
                for err in payload.get("errors") or ()
            ),
        )

    def station(self) -> Unit | None:
        for unit in self.ours:
            if unit.kind == STATION:
                return unit
        return None

    def enemy_station(self) -> Unit | None:
        for unit in self.enemy:
            if unit.kind == STATION:
                return unit
        return None

    def alive(self, kinds: tuple[str, ...]) -> tuple[Unit, ...]:
        return tuple(
            unit for unit in self.ours
            if unit.kind in kinds and unit.health > 0
        )

    def controllable(self) -> tuple[Unit, ...]:
        return tuple(sorted(
            self.alive(CONTROLLABLE_TYPES), key=lambda unit: unit.unit_id,
        ))

    def workers(self) -> tuple[Unit, ...]:
        return tuple(sorted(
            self.alive((WORKER,)), key=lambda unit: unit.unit_id,
        ))

    def pioneer(self) -> Unit | None:
        for unit in self.alive((PIONEER,)):
            return unit
        return None

    def weapons(self) -> tuple[Unit, ...]:
        return tuple(sorted(
            self.alive(TOWER_TYPES),
            key=lambda unit: (unit.pos.x, unit.pos.y),
        ))

    def walls(self) -> tuple[Unit, ...]:
        return self.alive((WALL,))

    def mines(self, kind: str) -> tuple[Pos, ...]:
        return tuple(
            pos for pos, zone in self.zones.items() if kind == zone
        )

    def stone_mines(self) -> tuple[Pos, ...]:
        return self.mines(WALL_MATERIAL)

    def zone_positions(self, kind: str) -> tuple[Pos, ...]:
        return self.mines(kind)

    def vendor(self) -> Pos | None:
        for pos, kind in self.zones.items():
            if kind == VENDOR:
                return pos
        return None

    def weapon_shop(self) -> Pos | None:
        for pos, kind in self.zones.items():
            if kind == WEAPON_SHOP:
                return pos
        return None

    def footprint(self, unit: Unit) -> tuple[Pos, ...]:
        if unit.kind == STATION:
            return station_footprint(unit.pos)
        return (unit.pos,)

    def land(self, pos: Pos) -> bool:
        if not 0 <= pos.x < self.width or not 0 <= pos.y < self.height:
            return False
        return self.zones.get(pos, LAND) == LAND

    def occupied_cells(self) -> frozenset[Pos]:
        cells: set[Pos] = set()
        for unit in self.ours:
            cells.update(self.footprint(unit))
        for unit in self.enemy:
            cells.update(self.footprint(unit))
        return frozenset(cells)

    def blocked(self, moving: Unit) -> frozenset[Pos]:
        cells = {pos for pos, kind in self.zones.items() if kind != LAND}
        cells.update(self.occupied_cells())
        cells.discard(moving.pos)
        for robot in self.robots:
            cells.add(robot.pos)
        return frozenset(cells)


def move_command(pos: Pos) -> dict[str, Any]:
    return {"action": "move", "targetPos": [pos.dump()]}


def collect_command(pos: Pos) -> dict[str, Any]:
    return {"action": "collect", "targetPos": [pos.dump()]}


def build_command(pos: Pos, name: str) -> dict[str, Any]:
    return {"action": "build", "targetPos": [pos.dump()], "name": name}


def remove_command(pos: Pos) -> dict[str, Any]:
    return {"action": "remove", "targetPos": [pos.dump()]}


def attack_command(controller_id: int, targets: list[Pos]) -> dict[str, Any]:
    return {
        "action": "attack",
        "targetPos": [pos.dump() for pos in targets],
        "controllerId": str(controller_id),
    }


def sell_command(name: str, num: int) -> dict[str, Any]:
    return {"action": "sell", "name": name, "num": int(num)}


def buy_command(name: str, num: int) -> dict[str, Any]:
    return {"action": "buy", "name": name, "num": int(num)}


def use_command(name: str, pos: Pos | None = None) -> dict[str, Any]:
    command: dict[str, Any] = {"action": "use", "name": name}
    if pos is not None:
        command["targetPos"] = [pos.dump()]
    return command


def drop_command(name: str) -> dict[str, Any]:
    return {"action": "drop", "name": name}


def accept_command() -> dict[str, Any]:
    return {"action": "acceptTask"}


def submit_command(answer: str) -> dict[str, Any]:
    return {"action": "submitAnswer", "taskAnswer": answer}


def summon_command(pos: Pos, items: list[str]) -> dict[str, Any]:
    return {
        "action": "summonTreasure",
        "targetPos": [pos.dump()],
        "item": list(items),
    }


@dataclass
class Response:
    commands: dict[int, dict[str, Any]] = field(default_factory=dict)
    prompt: str = ""
    execute_cmd: str = ""

    def dump(self) -> dict[str, Any]:
        return {
            "roleCommandMap": {
                str(key): value for key, value in self.commands.items()
            },
            "prompt": self.prompt,
            "executeCmd": self.execute_cmd,
        }
