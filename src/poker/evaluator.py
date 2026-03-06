from itertools import combinations
from .cards import Card, RANKS

RANK_TO_VAL = {r: i for i, r in enumerate(RANKS, start=2)}

# Hand categories (bigger is better):
# 8: straight flush, 7: quads, 6: full house, 5: flush, 4: straight, 3: trips, 2: two pair, 1: pair, 0: high card

def _is_straight(vals: list[int]) -> int | None:
    """Return high card of straight if straight else None. vals must be distinct."""
    vals = sorted(set(vals))
    # Wheel straight A-2-3-4-5
    if set([14, 2, 3, 4, 5]).issubset(vals):
        return 5
    for i in range(len(vals) - 4):
        window = vals[i:i+5]
        if window == list(range(window[0], window[0] + 5)):
            return window[-1]
    return None

def rank_five(cards: list[Card]) -> tuple[int, tuple[int, ...]]:
    """Return (category, tiebreak tuple). Higher is better."""
    suits = [c.suit for c in cards]
    vals = sorted([RANK_TO_VAL[c.rank] for c in cards], reverse=True)

    # counts
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)  # (val,count)

    is_flush = len(set(suits)) == 1
    straight_high = _is_straight([RANK_TO_VAL[c.rank] for c in cards])

    if is_flush and straight_high is not None:
        return (8, (straight_high,))
    if groups[0][1] == 4:
        quad = groups[0][0]
        kicker = max(v for v in vals if v != quad)
        return (7, (quad, kicker))
    if groups[0][1] == 3 and groups[1][1] == 2:
        trips = groups[0][0]
        pair = groups[1][0]
        return (6, (trips, pair))
    if is_flush:
        return (5, tuple(vals))
    if straight_high is not None:
        return (4, (straight_high,))
    if groups[0][1] == 3:
        trips = groups[0][0]
        kickers = sorted([v for v in vals if v != trips], reverse=True)
        return (3, (trips, *kickers))
    if groups[0][1] == 2 and groups[1][1] == 2:
        high_pair = max(groups[0][0], groups[1][0])
        low_pair = min(groups[0][0], groups[1][0])
        kicker = max(v for v in vals if v != high_pair and v != low_pair)
        return (2, (high_pair, low_pair, kicker))
    if groups[0][1] == 2:
        pair = groups[0][0]
        kickers = sorted([v for v in vals if v != pair], reverse=True)
        return (1, (pair, *kickers))
    return (0, tuple(vals))

def best_of_seven(cards: list[Card]) -> tuple[int, tuple[int, ...]]:
    if len(cards) != 7:
        raise ValueError("best_of_seven expects exactly 7 cards.")
    best = None
    for combo in combinations(cards, 5):
        r = rank_five(list(combo))
        if best is None or r > best:
            best = r
    assert best is not None
    return best