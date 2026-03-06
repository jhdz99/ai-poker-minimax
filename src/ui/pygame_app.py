import pygame
import os
from dataclasses import dataclass

from src.ui.assets import load_card_images
from src.game.engine import TexasHoldemEngine
from src.game.actions import ActionType, Action
from src.game.state import Street
from src.ai.agent import DeterminizedMinimaxAgent
from src.poker.evaluator import best_of_seven
from itertools import combinations
from src.poker.evaluator import rank_five, RANK_TO_VAL

@dataclass
class MovingCard:
    card_key: str
    face_up: bool
    start_pos: tuple[int, int]
    end_pos: tuple[int, int]
    start_ms: int
    duration_ms: int
    layer: str  # "p0", "p1", "board"

WIDTH, HEIGHT = 1100, 700

class PokerApp:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Poker - Determinized Minimax")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)

        self.engine = TexasHoldemEngine()
        self.state, self.deck = self.engine.new_hand()
        self.hand_settled = False
        self.last_winner = None  # 0,1,-1
        self.ai = DeterminizedMinimaxAgent()

        # Main action buttons
        self.btn_fold = pygame.Rect(60, 600, 140, 55)
        self.btn_call = pygame.Rect(220, 600, 180, 55)
        self.btn_raise = pygame.Rect(420, 600, 180, 55)

        # Raise sizing controls
        self.btn_minus = pygame.Rect(620, 600, 55, 55)
        self.btn_plus = pygame.Rect(680, 600, 55, 55)
        self.btn_allin = pygame.Rect(760, 600, 160, 55)

        # ---- Card rendering config ----
        self.card_size = (80, 116)
        cards_dir = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "cards")
        cards_dir = os.path.abspath(cards_dir)

        self.card_fronts, self.card_back = load_card_images(cards_dir, self.card_size)

        # Deck (source) position for animations
        self.deck_pos = (950, 90)

        # Animation queue
        self.moving_cards: list[MovingCard] = []
        self.overlay_alpha = 0  # for fade-in banners
        self.overlay_target = 0
        self.overlay_text = ("", "")  # (headline, reason)

        # Track dealt changes
        self._prev_board_len = len(self.state.board)
        self._prev_p0_hole = tuple(self.state.players[0].hole)
        self._prev_p1_hole = tuple(self.state.players[1].hole)

        # Raise sizing state (extra chips beyond call)
        self.raise_extra = self.engine.bb  # start at big blind as default step
        self.raise_step = self.engine.bb

    def _card_key(self, card) -> str:
        # Card is like rank='A', suit='s' etc.
        return f"{card.rank}{card.suit}"

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _ease_out(self, t: float) -> float:
        # quick easing curve for nicer animation
        return 1 - (1 - t) * (1 - t)

    def _enqueue_card(self, card_key: str, face_up: bool, start: tuple[int,int], end: tuple[int,int], layer: str, duration_ms: int = 220):
        self.moving_cards.append(
            MovingCard(
                card_key=card_key,
                face_up=face_up,
                start_pos=start,
                end_pos=end,
                start_ms=pygame.time.get_ticks(),
                duration_ms=duration_ms,
                layer=layer
            )
        )

    def _p0_card_positions(self) -> list[tuple[int,int]]:
        # bottom center-ish
        x0, y0 = 420, 360
        gap = 90
        return [(x0, y0), (x0 + gap, y0)]

    def _p1_card_positions(self) -> list[tuple[int,int]]:
        # top center-ish
        x0, y0 = 420, 140
        gap = 90
        return [(x0, y0), (x0 + gap, y0)]

    def _board_positions(self) -> list[tuple[int,int]]:
        x0, y0 = 280, 250
        gap = 90
        return [(x0 + i * gap, y0) for i in range(5)]

    def _sync_deal_animations(self):
        """
        Called each frame.
        If state has new hole cards or new board cards vs previous snapshot,
        queue animations from deck -> destination.
        """
        p0, p1 = self.state.players

        # Hole cards changes (new hand)
        if tuple(p0.hole) != self._prev_p0_hole:
            self._prev_p0_hole = tuple(p0.hole)
            dests = self._p0_card_positions()
            for i, c in enumerate(p0.hole):
                self._enqueue_card(self._card_key(c), True, self.deck_pos, dests[i], "p0", duration_ms=240 + i * 60)

        if tuple(p1.hole) != self._prev_p1_hole:
            self._prev_p1_hole = tuple(p1.hole)
            dests = self._p1_card_positions()
            for i, c in enumerate(p1.hole):
                # AI cards face-down until showdown
                face_up = (self.state.street == Street.SHOWDOWN)
                self._enqueue_card(self._card_key(c), face_up, self.deck_pos, dests[i], "p1", duration_ms=240 + i * 60)

        # Board cards added
        if len(self.state.board) != self._prev_board_len:
            old = self._prev_board_len
            new = len(self.state.board)
            self._prev_board_len = new
            dests = self._board_positions()
            for i in range(old, new):
                c = self.state.board[i]
                self._enqueue_card(self._card_key(c), True, self.deck_pos, dests[i], "board", duration_ms=220)

    def _update_overlay(self):
        # fade overlay in/out
        if self.overlay_alpha < self.overlay_target:
            self.overlay_alpha = min(self.overlay_target, self.overlay_alpha + 18)
        elif self.overlay_alpha > self.overlay_target:
            self.overlay_alpha = max(self.overlay_target, self.overlay_alpha - 18)

    def _show_overlay(self, headline: str, reason: str):
        self.overlay_text = (headline, reason)
        self.overlay_target = 210  # visible

    def _hide_overlay(self):
        self.overlay_target = 0

    def _is_game_over(self) -> bool:
        p0, p1 = self.state.players
        return p0.stack <= 0 or p1.stack <= 0

    def _start_new_session(self):
        # Reset both players to the engine's starting stacks
        self.state, self.deck = self.engine.new_hand(
            stacks=(self.engine.starting_stack, self.engine.starting_stack)
        )
        self.hand_settled = False
        self.last_winner = None
        # reset raise sizing for the new session
        self.raise_extra = self.engine.bb

    def _controls_enabled(self) -> bool:
        return (not self._is_game_over()
                and self.state.street != Street.SHOWDOWN
                and self.state.to_act == 0)

    def _to_call(self) -> int:
        p = self.state.players[0]
        return max(0, self.state.current_bet - p.bet)

    def _clamp_raise_extra(self):
        p = self.state.players[0]
        to_call = self._to_call()
        # Max extra is whatever remains after calling
        max_extra = max(0, p.stack - to_call)
        self.raise_extra = max(0, min(self.raise_extra, max_extra))

    def _best_five(self, seven_cards):
        """
        Returns: (best_rank_tuple, best_5_cards_list)
        best_rank_tuple is what rank_five returns: (category, tiebreak_tuple)
        """
        best_rank = None
        best_hand = None
        for combo in combinations(seven_cards, 5):
            r = rank_five(list(combo))
            if best_rank is None or r > best_rank:
                best_rank = r
                best_hand = list(combo)
        return best_rank, best_hand

    def _val_to_rank_char(self, v: int) -> str:
        # Inverse of RANK_TO_VAL
        for k, val in RANK_TO_VAL.items():
            if val == v:
                return k
        return "?"

    def _rank_name(self, v: int) -> str:
        # Poker-friendly names
        mapping = {
            14: "Ace", 13: "King", 12: "Queen", 11: "Jack", 10: "Ten",
            9: "Nine", 8: "Eight", 7: "Seven", 6: "Six", 5: "Five",
            4: "Four", 3: "Three", 2: "Two"
        }
        return mapping.get(v, str(v))
    
    def _hand_description_from_rank(self, rank_tuple) -> str:
        """
        rank_tuple: (category, tiebreak)
        Uses the tiebreak structure from your evaluator to produce a readable description.
        """
        category, t = rank_tuple
        # category mapping from evaluator.py comments:
        # 8 SF, 7 quads, 6 FH, 5 flush, 4 straight, 3 trips, 2 two pair, 1 pair, 0 high
        if category == 8:
            return f"Straight Flush ({self._rank_name(t[0])}-high)"
        if category == 7:
            return f"Four of a Kind ({self._rank_name(t[0])}s)"
        if category == 6:
            return f"Full House ({self._rank_name(t[0])}s full of {self._rank_name(t[1])}s)"
        if category == 5:
            return f"Flush ({self._rank_name(t[0])}-high)"
        if category == 4:
            return f"Straight ({self._rank_name(t[0])}-high)"
        if category == 3:
            return f"Three of a Kind ({self._rank_name(t[0])}s)"
        if category == 2:
            return f"Two Pair ({self._rank_name(t[0])}s and {self._rank_name(t[1])}s)"
        if category == 1:
            return f"One Pair ({self._rank_name(t[0])}s)"
        return f"High Card ({self._rank_name(t[0])})"
    
    def _showdown_message_and_reason(self) -> tuple[str, str]:
        """
        Returns (headline, reason)
        headline: "You Win!" / "You Lost!" / "Draw!"
        reason: a short explanation string
        """
        p0, p1 = self.state.players  # you, AI

        # Fold outcomes
        if p0.folded and not p1.folded:
            return "You Lost!", "Reason: You folded."
        if p1.folded and not p0.folded:
            return "You Win!", "Reason: AI folded."
        if p0.folded and p1.folded:
            return "Draw!", "Reason: Both folded (unexpected state)."

        # Need full board to evaluate properly
        if len(self.state.board) < 5:
            return "Showdown!", "Reason: Board is incomplete."

        your_best_rank, _ = self._best_five(p0.hole + self.state.board)
        ai_best_rank, _ = self._best_five(p1.hole + self.state.board)

        your_desc = self._hand_description_from_rank(your_best_rank)
        ai_desc = self._hand_description_from_rank(ai_best_rank)

        if your_best_rank > ai_best_rank:
            return "You Win!", f"Reason: Your {your_desc} beat AI's {ai_desc}."
        if your_best_rank < ai_best_rank:
            return "You Lost!", f"Reason: AI's {ai_desc} beat your {your_desc}."
        return "Draw!", f"Reason: Tie — you both had {your_desc}."
    
    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            # AI turn
            if self.state.street != Street.SHOWDOWN and self.state.to_act == 1:
                from src.ai.node import ImperfectInfoRoot

                root = ImperfectInfoRoot(
                    state=self.state,
                    remaining_deck_cards=list(self.deck.cards),  # unseen pool for determinization
                    engine=self.engine,
                    ai_index=1
                )
                action = self.ai.choose_action(root)
                self.state = self.engine.apply(self.state, self.deck, action)

            # If hand ended, pay winner ONCE
            if self.state.street == Street.SHOWDOWN and not self.hand_settled:
                self.last_winner = self.engine.settle_hand(self.state)
                self.hand_settled = True
                headline, reason = self._showdown_message_and_reason()
                self._show_overlay(headline, reason)

            self.draw()

            # queue animations if new cards appeared
            self._sync_deal_animations()

            # overlay fade update
            self._update_overlay()

        pygame.quit()

    def _basic_ai_fallback(self, legal):
        # placeholder: call/check unless raise exists sometimes
        raise_actions = [a for a in legal if a.type == ActionType.BET_RAISE]
        if raise_actions and pygame.time.get_ticks() % 2 == 0:
            return raise_actions[0]
        for a in legal:
            if a.type == ActionType.CHECK_CALL:
                return a
        return legal[0]

    def handle_click(self, pos):
        # If someone is busted, clicking starts a brand new session
        if self._is_game_over():
            self._start_new_session()
            return

        if self.state.street == Street.SHOWDOWN:
            # carry stacks forward
            stacks = (self.state.players[0].stack, self.state.players[1].stack)
            self.state, self.deck = self.engine.new_hand(stacks=stacks)
            self.hand_settled = False
            self.last_winner = None
            # reset raise sizing for the new hand
            self.raise_extra = self.engine.bb
            return
        
        self._hide_overlay()

        # If it's not your turn, ignore clicks
        if self.state.to_act != 0:
            return

        # Raise sizing controls
        if self.btn_minus.collidepoint(pos):
            self.raise_extra -= self.raise_step
            self._clamp_raise_extra()
            return

        if self.btn_plus.collidepoint(pos):
            self.raise_extra += self.raise_step
            self._clamp_raise_extra()
            return

        if self.btn_allin.collidepoint(pos):
            p = self.state.players[0]
            to_call = self._to_call()
            all_in_amount_now = p.stack  # put in everything remaining now
            # If you want "all-in" to always at least call first, you can keep it as p.stack.
            self.state = self.engine.apply(self.state, self.deck, Action(ActionType.BET_RAISE, amount=all_in_amount_now))
            return

        # Standard actions
        if self.btn_fold.collidepoint(pos):
            self.state = self.engine.apply(self.state, self.deck, Action(ActionType.FOLD))
        elif self.btn_call.collidepoint(pos):
            self.state = self.engine.apply(self.state, self.deck, Action(ActionType.CHECK_CALL))
        elif self.btn_raise.collidepoint(pos):
            p = self.state.players[0]
            to_call = self._to_call()
            self._clamp_raise_extra()

            # amount is what we put in NOW: call + extra
            amount_now = min(p.stack, to_call + self.raise_extra)
            self.state = self.engine.apply(self.state, self.deck, Action(ActionType.BET_RAISE, amount=amount_now))

    def draw(self):
        self.screen.fill((18, 120, 70))  # table green

        p0, p1 = self.state.players

        # Header text
        self._draw_text(f"Pot: {self.state.pot}", 60, 40)
        self._draw_text(f"Street: {self.state.street.name}", 60, 70)
        self._draw_text(f"You stack: {p0.stack}", 60, 120)
        self._draw_text(f"AI stack: {p1.stack}", 60, 150)

        # Draw deck (static)
        self.screen.blit(self.card_back, self.deck_pos)

        # ---- Draw static cards (not currently moving) ----
        moving_keys = set()
        for mc in self.moving_cards:
            moving_keys.add((mc.layer, mc.card_key))

        # Board
        board_dests = self._board_positions()
        for i, c in enumerate(self.state.board):
            key = self._card_key(c)
            if ("board", key) in moving_keys:
                continue
            self.screen.blit(self.card_fronts[key], board_dests[i])

        # Player 0 (you)
        p0_dests = self._p0_card_positions()
        for i, c in enumerate(p0.hole):
            key = self._card_key(c)
            if ("p0", key) in moving_keys:
                continue
            self.screen.blit(self.card_fronts[key], p0_dests[i])

        # Player 1 (AI)
        p1_dests = self._p1_card_positions()
        for i, c in enumerate(p1.hole):
            key = self._card_key(c)
            if ("p1", key) in moving_keys:
                continue
            face_up = (self.state.street == Street.SHOWDOWN)
            img = self.card_fronts[key] if face_up else self.card_back
            self.screen.blit(img, p1_dests[i])

        # ---- Animate moving cards on top ----
        now = pygame.time.get_ticks()
        still_moving: list[MovingCard] = []
        for mc in self.moving_cards:
            t = (now - mc.start_ms) / max(1, mc.duration_ms)
            if t >= 1.0:
                # animation finished -> allow it to become static next frame
                continue
            t = max(0.0, min(1.0, t))
            t = self._ease_out(t)

            x = int(self._lerp(mc.start_pos[0], mc.end_pos[0], t))
            y = int(self._lerp(mc.start_pos[1], mc.end_pos[1], t))

            img = self.card_fronts[mc.card_key] if mc.face_up else self.card_back
            self.screen.blit(img, (x, y))
            still_moving.append(mc)

        self.moving_cards = still_moving

        # ---- Controls + bet text (only while active hand) ----
        enabled = self._controls_enabled()
        btn_color = (40, 40, 40) if enabled else (90, 90, 90)

        pygame.draw.rect(self.screen, btn_color, self.btn_fold, border_radius=10)
        pygame.draw.rect(self.screen, btn_color, self.btn_call, border_radius=10)
        pygame.draw.rect(self.screen, btn_color, self.btn_raise, border_radius=10)
        self._draw_text("Fold", self.btn_fold.x + 45, self.btn_fold.y + 15)
        self._draw_text("Check/Call", self.btn_call.x + 35, self.btn_call.y + 15)
        self._draw_text("Raise", self.btn_raise.x + 60, self.btn_raise.y + 15)

        pygame.draw.rect(self.screen, btn_color, self.btn_minus, border_radius=10)
        pygame.draw.rect(self.screen, btn_color, self.btn_plus, border_radius=10)
        pygame.draw.rect(self.screen, btn_color, self.btn_allin, border_radius=10)
        self._draw_text("-", self.btn_minus.x + 22, self.btn_minus.y + 15)
        self._draw_text("+", self.btn_plus.x + 20, self.btn_plus.y + 15)
        self._draw_text("All In", self.btn_allin.x + 50, self.btn_allin.y + 15)

        to_call = self._to_call()
        self._clamp_raise_extra()
        total_now = min(p0.stack, to_call + self.raise_extra)
        self._draw_text(f"To call: {to_call}", 60, 460)
        self._draw_text(f"Raise extra: {self.raise_extra} (step {self.raise_step})", 60, 485)
        self._draw_text(f"Total put in now: {total_now}", 60, 510)

        # ---- Center overlay banner (fade-in) ----
        if self.overlay_alpha > 0:
            headline, reason = self.overlay_text

            overlay = pygame.Surface((WIDTH, 170))
            overlay.set_alpha(self.overlay_alpha)
            overlay.fill((20, 20, 20))
            self.screen.blit(overlay, (0, 430))

            # Bigger headline
            big = pygame.font.SysFont("arial", 44, bold=True)
            small = pygame.font.SysFont("arial", 22)

            # Color-code headline
            if "Win" in headline:
                color = (80, 220, 120)
            elif "Lost" in headline:
                color = (240, 90, 90)
            else:
                color = (240, 240, 240)

            head_surf = big.render(headline, True, color)
            reason_surf = small.render(reason, True, (240, 240, 240))
            hint_surf = small.render("Click to continue.", True, (240, 240, 240))

            # Center text
            hx = (WIDTH - head_surf.get_width()) // 2
            rx = (WIDTH - reason_surf.get_width()) // 2
            ix = (WIDTH - hint_surf.get_width()) // 2

            self.screen.blit(head_surf, (hx, 455))
            self.screen.blit(reason_surf, (rx, 515))
            self.screen.blit(hint_surf, (ix, 545))

        pygame.display.flip()

    def _draw_text(self, text, x, y):
        surf = self.font.render(text, True, (240, 240, 240))
        self.screen.blit(surf, (x, y))