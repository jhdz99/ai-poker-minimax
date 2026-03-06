from dataclasses import dataclass, field
from enum import Enum, auto
from src.poker.cards import Card

class Street(Enum):
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()

@dataclass
class PlayerState:
    stack: int
    bet: int = 0
    hole: list[Card] = field(default_factory=list)
    folded: bool = False

@dataclass
class GameState:
    pot: int
    board: list[Card]
    deck_exhausted: bool
    street: Street
    to_act: int  # 0 = human, 1 = AI
    players: list[PlayerState]
    current_bet: int = 0
    min_raise: int = 10

    def clone(self) -> "GameState":
        import copy
        return copy.deepcopy(self)