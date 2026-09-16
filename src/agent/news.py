import re

from .protocol import Turn
from .state import GameState

BASE_PRICE = {"stone": 1, "iron": 3, "copper": 5}
_ORE_WORDS = {
    "iron": ("铁", "铁矿"),
    "copper": ("铜", "铜矿"),
    "stone": ("石", "石矿", "采石"),
}
_BAD_EVENTS = ("停工", "塌方", "爆炸", "罢工", "检修", "事故", "封锁", "停产")
_GOOD_EVENTS = ("恢复", "复产", "复工", "重开", "解禁")
_BOOST = 1.6


def ingest(turn: Turn, state: GameState) -> None:
    if turn.day_round != 0:
        return
    for ore, price in turn.vendor_prices.items():
        state.price_history.setdefault(ore, []).append((turn.day, price))
    news = turn.official_news
    if news and (not state.news_seen or state.news_seen[-1][1] != news):
        state.news_seen.append((turn.day, news))
    _update_forecast(turn, state)


def _update_forecast(turn: Turn, state: GameState) -> None:
    forecast: dict[str, float] = {}
    for ore in BASE_PRICE:
        forecast[ore] = float(_baseline(state, ore))
    news = turn.official_news
    for ore, words in _ORE_WORDS.items():
        mentioned = any(word in news for word in words)
        if not mentioned:
            continue
        if any(word in news for word in _BAD_EVENTS):
            days = _extract_days(news)
            boost = _learned_boost(state, ore)
            horizon = turn.day + days
            for day in range(turn.day, horizon + 1):
                forecast[ore] = max(forecast[ore], _baseline(state, ore) * boost)
        elif any(word in news for word in _GOOD_EVENTS):
            forecast[ore] = float(_baseline(state, ore))
    state.price_forecast = forecast


def _baseline(state: GameState, ore: str) -> float:
    history = state.price_history.get(ore) or []
    if history:
        return float(history[-1][1])
    return float(BASE_PRICE[ore])


def _learned_boost(state: GameState, ore: str) -> float:
    history = state.price_history.get(ore) or []
    if len(history) < 2:
        return _BOOST
    base = BASE_PRICE[ore]
    spikes = [price / base for _day, price in history if price > base]
    if spikes:
        return max(_BOOST, sum(spikes) / len(spikes))
    return _BOOST


def _extract_days(news: str) -> int:
    match = re.search(r"(\d+)\s*天", news)
    if match:
        return int(match.group(1))
    return 2


def forecast(ore: str, state: GameState) -> float:
    return state.price_forecast.get(ore, float(BASE_PRICE[ore]))


def spiking(ore: str, state: GameState) -> bool:
    base = _baseline(state, ore)
    return forecast(ore, state) > base * 1.2
