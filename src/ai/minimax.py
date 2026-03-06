from math import inf
from src.game.actions import ActionType

def minimax(node, depth, alpha, beta, maximizing: bool):
    """
    node interface:
      - node.is_terminal() -> bool
      - node.utility() -> float (from AI perspective)
      - node.legal_actions() -> list[Action]
      - node.apply(action) -> new_node
    """
    if depth == 0 or node.is_terminal():
        return node.utility(), None

    best_action = None

    if maximizing:
        value = -inf
        for a in node.legal_actions():
            v, _ = minimax(node.apply(a), depth - 1, alpha, beta, False)
            if v > value:
                value = v
                best_action = a
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value, best_action
    else:
        value = inf
        for a in node.legal_actions():
            v, _ = minimax(node.apply(a), depth - 1, alpha, beta, True)
            if v < value:
                value = v
                best_action = a
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_action