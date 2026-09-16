from .protocol import Turn
from .state import GameState, LLM_DAILY_LIMIT

PURPOSE_LEGEND = "legend"
PURPOSE_TREASURE_FINAL = "treasure_final"
PURPOSE_TASK_PLAN = "task_plan"
PURPOSE_TASK_EXTRACT = "task_extract"


def in_task(state: GameState) -> bool:
    return state.task.active


def budget_ok(turn: Turn, state: GameState) -> bool:
    if in_task(state):
        return True
    if state.llm_disabled_today:
        return False
    if state.llm_day != turn.day:
        return True
    return state.llm_used_today < LLM_DAILY_LIMIT


def note_sent(turn: Turn, state: GameState) -> None:
    if in_task(state):
        return
    if state.llm_day != turn.day:
        state.llm_day = turn.day
        state.llm_used_today = 0
    state.llm_used_today += 1


def note_quota_error(state: GameState) -> None:
    state.llm_disabled_today = True
