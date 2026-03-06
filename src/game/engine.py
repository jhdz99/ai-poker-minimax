from src.poker.deck import Deck
from src.poker.cards import Card
from .state import GameState, PlayerState, Street
from .actions import Action, ActionType
from src.poker.evaluator import best_of_seven

class TexasHoldemEngine:
    def __init__(self, starting_stack: int = 1000, small_blind: int = 5, big_blind: int = 10):
        self.starting_stack = starting_stack
        self.sb = small_blind
        self.bb = big_blind

    def settle_hand(self, state: GameState) -> int:
        """
        Pays out the pot and returns:
        0 if human wins
        1 if AI wins
        -1 if tie
        This should be called exactly ONCE per hand when street == SHOWDOWN.
        """
        # include any bets that might still be sitting in bet fields
        total_pot = state.pot + sum(pl.bet for pl in state.players)
        state.pot = total_pot
        for pl in state.players:
            pl.bet = 0

        p0, p1 = state.players

        # fold outcomes
        if p0.folded and not p1.folded:
            p1.stack += total_pot
            state.pot = 0
            return 1
        if p1.folded and not p0.folded:
            p0.stack += total_pot
            state.pot = 0
            return 0

        # showdown outcomes
        # (your engine advances to SHOWDOWN after river; board should be 5 cards)
        if len(state.board) < 5:
            # safety: if somehow incomplete, split
            split = total_pot // 2
            p0.stack += split
            p1.stack += total_pot - split
            state.pot = 0
            return -1

        r0 = best_of_seven(p0.hole + state.board)
        r1 = best_of_seven(p1.hole + state.board)

        if r0 > r1:
            p0.stack += total_pot
            winner = 0
        elif r1 > r0:
            p1.stack += total_pot
            winner = 1
        else:
            # split pot (odd chip goes to AI here; you can choose otherwise)
            split = total_pot // 2
            p0.stack += split
            p1.stack += total_pot - split
            winner = -1

        state.pot = 0
        return winner

    def new_hand(self, stacks: tuple[int, int] | None = None) -> tuple[GameState, Deck]:
        deck = Deck()

        if stacks is None:
            s0 = self.starting_stack
            s1 = self.starting_stack
        else:
            s0, s1 = stacks

        p0 = PlayerState(stack=s0)
        p1 = PlayerState(stack=s1)

        p0.hole = deck.draw(2)
        p1.hole = deck.draw(2)

        # blinds (handle short stacks safely)
        sb_amt = min(p0.stack, self.sb)
        bb_amt = min(p1.stack, self.bb)

        p0.bet = sb_amt
        p0.stack -= sb_amt

        p1.bet = bb_amt
        p1.stack -= bb_amt

        state = GameState(
            pot=0,
            board=[],
            deck_exhausted=False,
            street=Street.PREFLOP,
            to_act=0,  # keep your current simplification
            players=[p0, p1],
            current_bet=bb_amt,
            min_raise=self.bb
        )

        self._move_bets_to_pot(state)
        return state, deck

    def legal_actions(self, state: GameState) -> list[Action]:
        p = state.players[state.to_act]
        if p.folded:
            return []

        to_call = max(0, state.current_bet - p.bet)
        actions: list[Action] = [Action(ActionType.FOLD), Action(ActionType.CHECK_CALL)]

        # If player can't even call, no raises
        if p.stack <= to_call:
            return actions

        # Raises: action.amount is "chips put in now"
        # We'll offer: min-raise, half-pot, pot, all-in
        pot = state.pot + sum(pl.bet for pl in state.players)

        candidates: set[int] = set()

        # 1) Minimum legal raise (call + min_raise)
        min_raise_now = to_call + state.min_raise
        if min_raise_now > to_call:
            candidates.add(min_raise_now)

        # 2) Half-pot and pot sizing (common poker sizes)
        # total put in now roughly = call + size
        half_pot_now = to_call + max(state.min_raise, pot // 2)
        pot_now = to_call + max(state.min_raise, pot)

        candidates.add(half_pot_now)
        candidates.add(pot_now)

        # 3) All-in
        candidates.add(p.stack)

        # Filter & clamp to stack, remove <= call
        cleaned = []
        for amt in candidates:
            amt = min(p.stack, int(amt))
            if amt > to_call:
                cleaned.append(amt)

        # Sort small -> big so AI can consider sizes consistently
        for amt in sorted(set(cleaned)):
            actions.append(Action(ActionType.BET_RAISE, amount=amt))

        return actions

    def apply(self, state: GameState, deck: Deck, action: Action) -> GameState:
        s = state.clone()
        p = s.players[s.to_act]
        opp = s.players[1 - s.to_act]

        if action.type == ActionType.FOLD:
            p.folded = True
            s.street = Street.SHOWDOWN
            return s

        to_call = max(0, s.current_bet - p.bet)

        if action.type == ActionType.CHECK_CALL:
            call_amt = min(p.stack, to_call)
            p.stack -= call_amt
            p.bet += call_amt      

        elif action.type == ActionType.BET_RAISE:
            # action.amount = chips to put in NOW (includes call portion)
            desired = max(0, int(action.amount))
            desired = min(desired, p.stack)

            to_call = max(0, s.current_bet - p.bet)

            # If they didn't even cover the call, treat it as a call
            if desired <= to_call:
                call_amt = min(p.stack, to_call)
                p.stack -= call_amt
                p.bet += call_amt
            else:
                # Enforce minimum raise if player is NOT all-in
                min_total = to_call + s.min_raise

                # If they can afford a legal raise but chose less, bump to min raise
                if desired < min_total and p.stack > min_total:
                    desired = min_total

                # If desired is all-in and it's less than min raise, allow it
                put_in = min(p.stack, desired)
                p.stack -= put_in
                p.bet += put_in
                s.current_bet = max(s.current_bet, p.bet)

        # if both have matched bets (or someone all-in), advance
        if self._bets_settled(s):
            self._move_bets_to_pot(s)
            self._advance_street(s, deck)
            s.current_bet = 0
            for pl in s.players:
                pl.bet = 0

        s.to_act = 1 - s.to_act
        return s

    def _bets_settled(self, s: GameState) -> bool:
        p0, p1 = s.players
        if p0.folded or p1.folded:
            return True
        # if both equal or someone can't continue (all-in)
        if p0.bet == p1.bet:
            return True
        if p0.stack == 0 or p1.stack == 0:
            return True
        return False

    def _move_bets_to_pot(self, s: GameState):
        for pl in s.players:
            s.pot += pl.bet
            pl.bet = 0

    def _advance_street(self, s: GameState, deck: Deck):
        if s.street == Street.PREFLOP:
            s.board.extend(deck.draw(3))
            s.street = Street.FLOP
        elif s.street == Street.FLOP:
            s.board.extend(deck.draw(1))
            s.street = Street.TURN
        elif s.street == Street.TURN:
            s.board.extend(deck.draw(1))
            s.street = Street.RIVER
        elif s.street == Street.RIVER:
            s.street = Street.SHOWDOWN