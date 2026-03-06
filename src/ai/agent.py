# src/ai/agent.py
import random
from dataclasses import dataclass
from src.game.actions import Action
from src.ai.node import ImperfectInfoRoot

@dataclass
class AIDecisionConfig:
    depth: int = 3                 # 2–3 is a good portfolio sweet spot
    determinizations: int = 10     # more = stronger, slower
    rollout_samples: int = 0       # not used in deterministic leaf eval (kept for extensibility)

class DeterminizedMinimaxAgent:
    def __init__(self, config: AIDecisionConfig | None = None, rng=None):
        self.cfg = config or AIDecisionConfig()
        self.rng = rng or random.Random()

    def choose_action(self, root: ImperfectInfoRoot) -> Action:
        actions = root.legal_actions()
        if not actions:
            raise RuntimeError("No legal actions for AI.")

        best_action = actions[0]
        best_score = float("-inf")

        for a in actions:
            score = 0.0
            for _ in range(self.cfg.determinizations):
                score += root.score_action_with_determinization(
                    a,
                    depth=self.cfg.depth,
                    rollout_samples=self.cfg.rollout_samples,
                    rng=self.rng
                )
            score /= self.cfg.determinizations

            if score > best_score:
                best_score = score
                best_action = a

        return best_action