import random
from .cards import Card, RANKS, SUITS

class Deck:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.cards = [Card(r, s) for r in RANKS for s in SUITS]
        self.shuffle()

    def shuffle(self):
        self.rng.shuffle(self.cards)

    def draw(self, n: int = 1):
        if n < 1:
            return []
        if len(self.cards) < n:
            raise RuntimeError("Deck is empty.")
        out = self.cards[:n]
        self.cards = self.cards[n:]
        return out