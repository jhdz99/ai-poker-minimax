import random
from src.poker.deck import Deck
from src.poker.evaluator import best_of_seven

def estimate_win_prob(my_hole, board, unseen_cards, samples: int = 200, rng=None) -> float:
    rng = rng or random.Random()
    wins = ties = 0

    # unseen_cards: list[Card] that could still be in opponent hand / future board
    for _ in range(samples):
        pool = unseen_cards[:]
        rng.shuffle(pool)

        opp_hole = pool[:2]
        remaining = pool[2:]

        needed = 5 - len(board)
        runout = remaining[:needed]
        full_board = board + runout

        my_rank = best_of_seven(my_hole + full_board)
        opp_rank = best_of_seven(opp_hole + full_board)

        if my_rank > opp_rank:
            wins += 1
        elif my_rank == opp_rank:
            ties += 1

    return (wins + 0.5 * ties) / samples