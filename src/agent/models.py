"""请求/响应数据模型。

全部用 dataclass 表达，``from_dict`` 对缺失字段容错，避免判题器某字段缺省导致解析崩溃。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Pos:
    x: int = 0
    y: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "Pos":
        if not d:
            return cls()
        return cls(int(d.get("x", 0)), int(d.get("y", 0)))

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    def key(self) -> tuple[int, int]:
        return (self.x, self.y)

    def __hash__(self) -> int:
        return hash(self.key())


@dataclass
class Zone:
    pos: Pos = field(default_factory=Pos)
    neutral_type: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> "Zone":
        if not d:
            return cls()
        return cls(Pos.from_dict(d.get("pos")), d.get("neutralType", ""))


@dataclass
class Role:
    id: int = 0
    pos: Pos = field(default_factory=Pos)
    role_type: str = ""
    health: int = 0
    attack_power: int = 0
    attack_range: int = 0
    back_pack_capability: int = 0
    backpack: list[str] = field(default_factory=list)
    level: int = 1
    cooldown: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "Role":
        if not d:
            return cls()
        return cls(
            id=int(d.get("id", 0)),
            pos=Pos.from_dict(d.get("pos")),
            role_type=d.get("roleType", ""),
            health=int(d.get("health", 0)),
            attack_power=int(d.get("attackPower", 0)),
            attack_range=int(d.get("attackRange", 0)),
            back_pack_capability=int(d.get("backPackCapability", 0)),
            backpack=list(d.get("backpack") or []),
            level=int(d.get("level", 1)),
            cooldown=int(d.get("cooldown", 0)),
        )


@dataclass
class Robot:
    id: int = 0
    pos: Pos = field(default_factory=Pos)
    role_type: str = ""
    health: int = 0
    abnormal_state: str = ""
    target_team: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> "Robot":
        if not d:
            return cls()
        return cls(
            id=int(d.get("id", 0)),
            pos=Pos.from_dict(d.get("pos")),
            role_type=d.get("roleType", ""),
            health=int(d.get("health", 0)),
            abnormal_state=d.get("abnormalState", ""),
            target_team=d.get("targetTeam", ""),
        )


@dataclass
class PlayerTask:
    task_type: str = ""
    task_position: Pos = field(default_factory=Pos)
    cold_down_rounds: int = 0
    score_reward: int = 0
    gold_reward: int = 0
    is_valid: bool = False
    timeout_rounds: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "PlayerTask":
        if not d:
            return cls()
        return cls(
            task_type=d.get("taskType", ""),
            task_position=Pos.from_dict(d.get("taskPosition")),
            cold_down_rounds=int(d.get("coldDownRounds", 0)),
            score_reward=int(d.get("scoreReward", 0)),
            gold_reward=int(d.get("goldReward", 0)),
            is_valid=bool(d.get("isValid", False)),
            timeout_rounds=int(d.get("timeoutRounds", 0)),
        )


@dataclass
class ShopItem:
    name: str = ""
    price: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "ShopItem":
        if not d:
            return cls()
        return cls(name=d.get("name", ""), price=int(d.get("price", 0)))
