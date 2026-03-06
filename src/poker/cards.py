from dataclasses import dataclass

RANKS = "23456789TJQKA"
SUITS = "shdc"  # spades hearts diamonds clubs

@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    @staticmethod
    def from_str(s: str) -> "Card":
        s = s.strip()
        if len(s) != 2 or s[0] not in RANKS or s[1] not in SUITS:
            raise ValueError(f"Invalid card: {s}")
        return Card(rank=s[0], suit=s[1])