def evaluate_state_simple(win_prob: float, pot: int, to_call: int) -> float:
    # Expected value of calling (very simplified):
    # EV ≈ win_prob * pot - (1 - win_prob) * to_call
    return win_prob * pot - (1.0 - win_prob) * to_call