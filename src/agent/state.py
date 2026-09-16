from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import Pos, Turn

LLM_DAILY_LIMIT = 3
SUMMON_ORDER_DAILY_LIMIT = 10
FAIL_BLACKLIST_ROUNDS = 10
FAIL_THRESHOLD = 2


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
    llm_extract_requested: bool = False
    llm_extract_applied: bool = False
    submits: int = 0
    submitted: set[str] = field(default_factory=set)

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
