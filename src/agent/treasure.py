import json
import re

from .protocol import Pos, Turn, distance, move_command, summon_command
from .state import GameState, TreasureCase

ITEM_ALIASES = {
    "AcientTablet": ("古符石板", "石板", "tablet"),
    "StarSand": ("星辰之沙", "星沙", "沙", "sand"),
    "FlameBreath": ("烈焰之息", "烈焰", "火焰", "flame"),
    "FrostPotion": ("寒霜药剂", "寒霜", "冰霜", "frost"),
    "ThornAmulet": ("荆棘护符", "荆棘", "护符", "thorn"),
    "IronWhistle": ("回音铁哨", "铁哨", "哨子", "whistle"),
}
_DIRECTIONS = {
    "西": (-1, 0), "东": (1, 0), "南": (0, -1), "北": (0, 1),
    "左上": (-1, 1), "右上": (1, 1), "左下": (-1, -1), "右下": (1, -1),
}
_COORD_RE = re.compile(r"\(?\s*(\d{1,2})\s*[,，、]\s*(\d{1,2})\s*\)?")
_DAY_RE = re.compile(r"第\s*(\d+)\s*天")


def ingest_legend(turn: Turn, state: GameState) -> None:
    legend = turn.folk_legend
    if not legend:
        return
    case = state.treasure
    if case.done:
        return
    if case.legends and case.legends[-1] == (turn.day, legend):
        return
    if any(text == legend for _day, text in case.legends):
        return
    case.legends.append((turn.day, legend))
    _rule_extract(turn, state)


def _rule_extract(turn: Turn, state: GameState) -> None:
    case = state.treasure
    text = "\n".join(legend for _day, legend in case.legends)
    explicit: list[Pos] = []
    for match in _COORD_RE.finditer(text):
        x, y = int(match.group(1)), int(match.group(2))
        if 0 <= x < turn.width and 0 <= y < turn.height:
            pos = Pos(x, y)
            if turn.land(pos) and pos not in explicit:
                explicit.append(pos)
    if explicit:
        # 显式坐标优先，排在弱猜测之前
        rest = [pos for pos in case.loc_candidates if pos not in explicit]
        case.loc_candidates = explicit + rest
    elif not case.loc_candidates:
        guessed = _direction_guess(turn, text)
        if guessed is not None:
            case.loc_candidates.append(guessed)
    if not case.item_candidates:
        found = [
            item for item, aliases in ITEM_ALIASES.items()
            if any(alias in text for alias in aliases)
        ]
        if found:
            case.item_candidates.append(tuple(sorted(found)))
        number = _key_count(text)
        if number and len(found) > number:
            case.item_candidates = [tuple(sorted(found)[:number])]
    if case.open_day is None:
        match = _DAY_RE.search(text)
        if match:
            case.open_day = int(match.group(1))


def _key_count(text: str) -> int | None:
    match = re.search(r"([三3])\s*钥", text)
    if match:
        return 3
    match = re.search(r"([两2二])\s*钥", text)
    if match:
        return 2
    return None


def _direction_guess(turn: Turn, text: str) -> Pos | None:
    station = turn.station()
    if station is None:
        return None
    for word, (dx, dy) in _DIRECTIONS.items():
        if f"{word}部" in text or f"{word}边" in text or f"{word}侧" in text:
            anchor = station.pos
            if dx < 0:
                x = 4
            elif dx > 0:
                x = turn.width - 5
            else:
                x = anchor.x
            if dy < 0:
                y = 4
            elif dy > 0:
                y = turn.height - 5
            else:
                y = anchor.y
            pos = Pos(min(max(x, 1), turn.width - 2), min(max(y, 1), turn.height - 2))
            if turn.land(pos):
                return pos
    return None


def legend_prompt(state: GameState) -> str:
    case = state.treasure
    text = "\n".join(
        f"DAY{day}: {legend}" for day, legend in case.legends
    )
    items = "、".join(
        f"{name}({','.join(aliases[:2])})"
        for name, aliases in ITEM_ALIASES.items()
    )
    return (
        "你在一个41x32的游戏地图中寻宝，坐标原点(0,0)在左下角，x向右，y向上。"
        "以下是多天收集到的民间传闻：\n"
        f"{text}\n"
        f"可献祭的任务用品只有这些：{items}\n"
        "请推断宝藏的：1)祭坛坐标x,y 2)所需献祭物品(从列表中选，不能多不能少) "
        "3)开启时间(第几天)。只输出JSON："
        '{"x":int,"y":int,"items":[str],"day":int}'
    )


def apply_llm_answer(state: GameState, text: str) -> bool:
    case = state.treasure
    try:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return False
        data = json.loads(match.group(0))
        x, y = int(data["x"]), int(data["y"])
        pos = Pos(x, y)
        if pos not in case.loc_candidates:
            case.loc_candidates.insert(0, pos)
        items = tuple(
            sorted(str(item) for item in data.get("items") or ()
                   if str(item) in ITEM_ALIASES)
        )
        if items and items not in case.item_candidates:
            case.item_candidates.insert(0, items)
        day = data.get("day")
        if day:
            case.open_day = int(day)
        return True
    except Exception:
        return False


def ready(case: TreasureCase, turn: Turn) -> bool:
    if case.done or not case.loc_candidates or not case.item_candidates:
        return False
    if case.open_day is not None and turn.day < case.open_day:
        return False
    return True


def pioneer_action(turn: Turn, state: GameState) -> dict | None:
    from .grid import step_adjacent

    case = state.treasure
    pioneer = turn.pioneer()
    if pioneer is None or not ready(case, turn):
        return None
    target = case.loc_candidates[0]
    items = list(case.item_candidates[0])
    if any(not pioneer.has(item) for item in items):
        return None
    if distance(pioneer.pos, target) <= 1:
        return summon_command(target, items)
    step = step_adjacent(turn, pioneer, target)
    if step is not None:
        return move_command(step)
    return None


def handle_result(turn: Turn, state: GameState) -> None:
    case = state.treasure
    result = turn.last_summon
    if result == 0:
        return
    if result in (1, 4):
        case.done = True
        return
    if not case.loc_candidates or not case.item_candidates:
        return
    tried = (case.loc_candidates[0], case.item_candidates[0])
    case.tried.append(tried)
    if result == 3:
        case.item_candidates.append(case.item_candidates.pop(0))
    elif result == 2:
        if len(case.loc_candidates) > 1:
            case.loc_candidates.append(case.loc_candidates.pop(0))
        elif case.open_day is None:
            case.open_day = turn.day + 1
