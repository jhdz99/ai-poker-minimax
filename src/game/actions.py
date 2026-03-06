from dataclasses import dataclass
from enum import Enum, auto

class ActionType(Enum):
    FOLD = auto()
    CHECK_CALL = auto()
    BET_RAISE = auto()

@dataclass(frozen=True)
class Action:
    type: ActionType
    amount: int = 0  # for BET_RAISE