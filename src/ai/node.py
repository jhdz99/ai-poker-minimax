# src/ai/node.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import random

from src.game.state import GameState, Street
from src.game.actions import Action
from src.poker.cards import Card
from src.poker.evaluator import best_of_seven


class FixedDeck:
    """
    Determinized deck: draw() pops from a fixed, pre-shuffled list.
    This resolves chance nodes (future community cards) for minimax.
    """
    def __init__(self, cards_in_order: List[Card]):
        self.cards = list(cards_in_order)

    def draw(self, n: int = 1) -> List[Card]:
        if n < 1:
            return []
        if len(self.cards) < n:
            raise RuntimeError("FixedDeck ran out of cards.")
        out = self.cards[:n]
        self.cards = self.cards[n:]
        return out


@dataclass
class DeterminizedNode:
    """
    A perfect-information node for minimax:
      - opponent hole cards are fixed
      - remaining deck order is fixed
    """
    state: GameState
    deck: FixedDeck
    engine: any
    ai_index: int = 1  # AI is player 1 by default

    def is_terminal(self) -> bool:
        if self.state.street == Street.SHOWDOWN:
            return True
        p0, p1 = self.state.players
        return p0.folded or p1.folded

    def legal_actions(self) -> list[Action]:
        return self.engine.legal_actions(self.state)

    def apply(self, action: Action) -> "DeterminizedNode":
        new_state = self.engine.apply(self.state, self.deck, action)
        return DeterminizedNode(new_state, self.deck, self.engine, self.ai_index)

    def utility(self) -> float:
        """
        Utility from AI perspective = expected FINAL AI stack after this hand ends.
        This makes betting "cost" something (because stack decreases when you bet),
        and stops minimax from preferring all-in lines constantly.
        """
        total_pot = self.state.pot + sum(pl.bet for pl in self.state.players)

        ai = self.state.players[self.ai_index]
        opp = self.state.players[1 - self.ai_index]

        # If the AI folded, it cannot win the pot; its value is just its remaining stack.
        if ai.folded:
            return float(ai.stack)

        # If opponent folded, AI wins the pot.
        if opp.folded:
            return float(ai.stack + total_pot)

        # If not enough board cards yet, complete deterministically using fixed deck order.
        board = list(self.state.board)
        if len(board) < 5:
            need = 5 - len(board)
            remaining = list(self.deck.cards)
            if len(remaining) < need:
                # Safety fallback: assume a split
                return float(ai.stack + total_pot * 0.5)
            board = board + remaining[:need]

        ai_rank = best_of_seven(ai.hole + board)
        opp_rank = best_of_seven(opp.hole + board)

        if ai_rank > opp_rank:
            return float(ai.stack + total_pot)
        if ai_rank == opp_rank:
            return float(ai.stack + total_pot * 0.5)
        return float(ai.stack)


class ImperfectInfoRoot:
    """
    Root wrapper used by the agent:
    - does NOT know opponent hole cards
    - does NOT know remaining deck order
    It scores actions by sampling determinizations and running minimax on each.
    """
    def __init__(self, state: GameState, remaining_deck_cards: List[Card], engine, ai_index: int = 1):
        self.state = state
        self.remaining = list(remaining_deck_cards)  # unseen to the AI
        self.engine = engine
        self.ai_index = ai_index

    def legal_actions(self) -> list[Action]:
        return self.engine.legal_actions(self.state)

    def score_action_with_determinization(
        self,
        action: Action,
        depth: int,
        rollout_samples: int = 0,  # kept for compatibility; not used in deterministic leaf eval
        rng: Optional[random.Random] = None
    ) -> float:
        """
        One determinization:
          1) sample opponent hole cards from unseen pool
          2) fix the remaining deck order (chance resolution)
          3) apply candidate action
          4) run minimax + alpha-beta for (depth-1)
        """
        rng = rng or random.Random()

        # Must have at least 2 cards to sample opponent hole
        if len(self.remaining) < 2:
            return 0.0

        pool = list(self.remaining)
        rng.shuffle(pool)

        opp_hole = pool[:2]
        rest = pool[2:]
        rng.shuffle(rest)
        fixed_deck = FixedDeck(rest)

        # Clone base state and inject sampled opponent hole cards
        s = self.state.clone()
        s.players[1 - self.ai_index].hole = list(opp_hole)

        root = DeterminizedNode(state=s, deck=fixed_deck, engine=self.engine, ai_index=self.ai_index)

        # Apply the candidate action at the root (AI is choosing now)
        after = root.apply(action)

        # Next actor is opponent, so minimax is minimizing after AI's move
        from src.ai.minimax import minimax
        value, _ = minimax(after, depth=max(depth - 1, 0), alpha=float("-inf"), beta=float("inf"), maximizing=False)
        return float(value)