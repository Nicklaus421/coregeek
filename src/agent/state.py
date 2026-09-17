from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import Pos, Turn, distance

LLM_DAILY_LIMIT = 3
SUMMON_ORDER_DAILY_LIMIT = 10
FAIL_BLACKLIST_ROUNDS = 10
FAIL_THRESHOLD = 2
ATTACK_WINDOW = 18  # 只统计基地附近该距离内的机器人，远处的不代表进攻方向
ATTACK_NEAR = 9


@dataclass
class TaskSession:
    active: bool = False
    point_pos: Pos | None = None
    text: str = ""
    accept_round: int = 0
    timeout_rounds: int = 0
    step: str = "IDLE"  # IDLE/GOTO/ACCEPTED/EXPLORING/SUBMITTING
    transcript: list[tuple[str, str]] = field(default_factory=list)
    pending_cmds: list[str] = field(default_factory=list)
    last_issued: str = ""
    draft_answer: str = ""
    llm_plan_requested: bool = False
    llm_plan_applied: bool = False
    llm_plan_at: int = 0
    llm_extract_requested: bool = False
    llm_extract_applied: bool = False
    submits: int = 0
    submitted: set[str] = field(default_factory=set)
    bad_answers: set[str] = field(default_factory=set)
    searched_files: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.__dict__.update(TaskSession().__dict__)


@dataclass
class TreasureCase:
    legends: list[tuple[int, str]] = field(default_factory=list)
    loc_candidates: list[Pos] = field(default_factory=list)
    item_candidates: list[tuple[str, ...]] = field(default_factory=list)
    open_day: int | None = None
    tried: list[tuple[Pos, tuple[str, ...]]] = field(default_factory=list)
    done: bool = False
    llm_requested_day: int = 0

    def reset_after_swap(self) -> None:
        self.loc_candidates = []
        self.open_day = None
        self.tried = []
        self.llm_requested_day = 0


@dataclass
class GameState:
    last_round: int = 0
    my_side: str = ""
    day: int = 0
    price_history: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    price_forecast: dict[str, float] = field(default_factory=dict)
    news_seen: list[tuple[int, str]] = field(default_factory=list)
    llm_used_today: int = 0
    llm_day: int = 0
    llm_disabled_today: bool = False
    summon_orders_today: int = 0
    shopping_plan: list[str] = field(default_factory=list)
    task: TaskSession = field(default_factory=TaskSession)
    treasure: TreasureCase = field(default_factory=TreasureCase)
    sop_cache: dict[str, tuple[list[str], str]] = field(default_factory=dict)
    failed: dict[tuple[int, str], tuple[int, int]] = field(default_factory=dict)
    last_cmds: dict[int, str] = field(default_factory=dict)
    attack_sectors: dict[tuple[int, int], int] = field(default_factory=dict)
    wall_entrance: Pos | None = None
    mine_target: dict[int, Pos] = field(default_factory=dict)  # 单位 -> 认领的矿点
    mine_used: dict[Pos, int] = field(default_factory=dict)  # 矿点 -> 我方已采次数
    shop_run: set[int] = field(default_factory=set)  # 正在跑"卖货+采购"行程的单位

    def reset_all(self) -> None:
        legends = self.treasure.legends
        sop = self.sop_cache
        history = self.price_history
        self.__dict__.update(GameState().__dict__)
        self.treasure.legends = legends
        self.sop_cache = sop
        self.price_history = history


STATE = GameState()


def maybe_reset(turn: Turn) -> GameState:
    if STATE.last_round and turn.round_no <= STATE.last_round:
        STATE.reset_all()
    if STATE.my_side and turn.my_side != STATE.my_side:
        STATE.reset_all()
    STATE.last_round = turn.round_no
    STATE.my_side = turn.my_side
    if turn.day != STATE.day:
        STATE.day = turn.day
        STATE.llm_used_today = 0
        STATE.llm_disabled_today = False
        STATE.summon_orders_today = 0
    return STATE


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def record_attack(turn: Turn, state: GameState) -> None:
    """夜里记录机器人相对基地的来向，供白天决定修墙/建塔的位置。"""
    station = turn.station()
    if station is None:
        return
    for robot in turn.robots:
        if robot.health <= 0:
            continue
        # 只统计冲我方基地来的机器人；盯着对面基地的不代表我方受敌方向
        if robot.target_team and robot.target_team != state.my_side:
            continue
        span = distance(robot.pos, station.pos)
        if span > ATTACK_WINDOW:
            continue
        key = (_sign(robot.pos.x - station.pos.x), _sign(robot.pos.y - station.pos.y))
        if key == (0, 0):
            continue
        # 越靠近基地越能代表真实进攻方向
        weight = (3 if span <= ATTACK_NEAR else 1)
        state.attack_sectors[key] = state.attack_sectors.get(key, 0) + weight


def attack_bearing(state: GameState) -> tuple[float, float] | None:
    """观测到的主要进攻方向（单位向量），无数据时返回 None。"""
    if not state.attack_sectors:
        return None
    x_axis = sum(dx * count for (dx, _dy), count in state.attack_sectors.items())
    y_axis = sum(dy * count for (_dx, dy), count in state.attack_sectors.items())
    if x_axis == 0 and y_axis == 0:
        return None
    norm = (x_axis * x_axis + y_axis * y_axis) ** 0.5
    return (x_axis / norm, y_axis / norm)


def unit_vector(dx: float, dy: float) -> tuple[float, float] | None:
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
        return None
    return (dx / norm, dy / norm)


def attack_direction(turn: Turn, state: GameState) -> tuple[float, float] | None:
    """进攻方向：夜间实测 > 敌方基地方向 > 指向地图内部（远离我方最近边缘）。"""
    observed = attack_bearing(state)
    if observed is not None:
        return observed
    station = turn.station()
    base = station.pos if station is not None else None
    if base is None:
        return None
    enemy = turn.enemy_station()
    if enemy is not None:
        inward = unit_vector(
            enemy.pos.x - base.x, enemy.pos.y - base.y,
        )
        if inward is not None:
            return inward
    return unit_vector(
        (turn.width - 1) / 2 - base.x,
        (turn.height - 1) / 2 - base.y,
    )


def facing_score(
    pos: Pos, origin: Pos, bearing: tuple[float, float] | None,
) -> float:
    """1 表示正对进攻方向，-1 表示完全背向。"""
    if bearing is None:
        return 0.0
    dx = pos.x - origin.x
    dy = pos.y - origin.y
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
        return 0.0
    return (dx * bearing[0] + dy * bearing[1]) / norm


def fingerprint(unit_id: int, cmd: dict) -> str:
    action = str(cmd.get("action"))
    targets = cmd.get("targetPos") or []
    coords = ",".join(f"{p.get('x')}:{p.get('y')}" for p in targets)
    return f"{action}|{cmd.get('name', '')}|{coords}"


def record_results(turn: Turn, state: GameState) -> None:
    for unit_id, ok in turn.last_results.items():
        fp = state.last_cmds.get(unit_id)
        if fp is None:
            continue
        key = (unit_id, fp)
        count, _round = state.failed.get(key, (0, 0))
        if ok:
            state.failed.pop(key, None)
        else:
            state.failed[key] = (count + 1, turn.round_no)


def blacklisted(unit_id: int, cmd: dict, state: GameState, round_no: int) -> bool:
    fp = fingerprint(unit_id, cmd)
    entry = state.failed.get((unit_id, fp))
    if entry is None:
        return False
    count, last_round = entry
    if count >= FAIL_THRESHOLD and round_no - last_round <= FAIL_BLACKLIST_ROUNDS:
        return True
    return False


def remember_cmds(state: GameState, commands: dict[int, dict]) -> None:
    state.last_cmds = {
        unit_id: fingerprint(unit_id, cmd) for unit_id, cmd in commands.items()
    }
