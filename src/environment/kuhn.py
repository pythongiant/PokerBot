"""
Minimal poker environments: Kuhn poker and simplified Leduc poker.

Kuhn Poker: 3 card deck {J, Q, K}, 2 players, 1 betting round.
This serves as the primary test environment.

Partially Observable Markov Game (POMG) implementation with clear
state encoding for Transformer belief learning.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import numpy as np


class Action(Enum):
    """Poker actions."""
    FOLD = 0
    CALL = 1
    RAISE = 2
    CHECK = 3


@dataclass
class GameState:
    """Encodes full game state (used only for simulation, not visible to agent)."""
    
    # Public cards and betting history visible to all
    public_cards: List[int]  # In Kuhn: empty; in Leduc: visible community cards
    public_bets: List[List[Tuple[Action, int]]]  # Betting per street per player
    
    # Private cards (one per player)
    private_cards: List[int]  # indexed by player
    
    # Current player to act
    current_player: int
    
    # Chip stacks per player
    stacks: List[int]
    
    # Money in pot (committed chips)
    pot: int
    
    # Current street (0 = preflop, 1 = flop in Leduc)
    street: int
    
    # Is game terminal
    is_terminal: bool
    
    # Terminal payoffs (only if terminal)
    payoffs: Optional[List[float]] = None


@dataclass
class ObservableState:
    """Observation visible to agent: no opponent private cards."""
    
    # Public community cards
    public_cards: List[int]
    
    # Own private card
    own_card: int
    
    # Betting history (actions only, not amounts in simplified version)
    action_history: List[Tuple[int, Action, int]]  # (player, action, amount)
    
    # Current player to act
    current_player: int
    
    # Stack sizes
    stacks: List[int]
    
    # Current pot
    pot: int
    
    # Current street
    street: int


class KuhnPoker:
    """
    Kuhn poker implementation: 3-card deck, 2 players, single betting round.
    
    Canonical model for POMG research. Easy to solve via CFR baseline.
    """
    
    DECK = [0, 1, 2]  # J, Q, K cards
    NUM_PLAYERS = 2
    NUM_ACTIONS = 4  # FOLD, CALL, RAISE, CHECK
    
    def __init__(self, initial_stack: int = 100, ante: int = 1, 
                 max_raises: int = 4, seed: Optional[int] = None):
        """
        Initialize Kuhn poker environment.
        
        Args:
            initial_stack: Starting chips per player
            ante: Ante to be posted
            max_raises: Max raises per street (simplification)
            seed: Random seed
        """
        self.initial_stack = initial_stack
        self.ante = ante
        self.max_raises = max_raises
        self.rng = np.random.RandomState(seed if seed is not None else 42)
        
        self.reset()
    
    def reset(self) -> Tuple[GameState, ObservableState]:
        """
        Reset game to initial state.
        
        Returns:
            (full_state, observable_state)
        """
        # Deal cards: shuffle deck and assign
        cards = self.rng.permutation(self.DECK)
        
        # Post antes: deduct from each player's stack and add to pot
        initial_stacks = [self.initial_stack - self.ante, self.initial_stack - self.ante]
        initial_pot = 2 * self.ante
        
        # Record ante posting in betting history for completeness
        ante_bets = [[], []]  # Empty for now - could record as special ANTE actions
        
        # Each player gets one card, we'll have a "community" card unused
        # (standard Kuhn is 3 cards, 2 players, 1 unused)
        self.game_state = GameState(
            public_cards=[],
            public_bets=ante_bets,  # Two players
            private_cards=[cards[0], cards[1]],
            current_player=0,  # Player 0 acts first
            stacks=initial_stacks,
            pot=initial_pot,
            street=0,
            is_terminal=False,
            payoffs=None,
        )
        
        return self.game_state, self._get_observable_state(0)
    
    def _get_observable_state(self, player_id: int) -> ObservableState:
        """Get observation from player's perspective (no opponent cards)."""
        return ObservableState(
            public_cards=self.game_state.public_cards,
            own_card=self.game_state.private_cards[player_id],
            action_history=self._get_action_history(),
            current_player=self.game_state.current_player,
            stacks=self.game_state.stacks.copy(),
            pot=self.game_state.pot,
            street=self.game_state.street,
        )
    
    def _get_action_history(self) -> List[Tuple[int, Action, int]]:
        """Flatten action history for current street."""
        history = []
        for player_id, actions in enumerate(self.game_state.public_bets):
            for action, amount in actions:
                history.append((player_id, action, amount))
        return history
    
    def get_legal_actions(self, player_id: int) -> List[Action]:
        """
        Get legal actions for current player.
        
        Rules:
        - If all-in or can't bet: only CALL/FOLD
        - If no one bet yet: CHECK or RAISE
        - If someone bet: FOLD, CALL, or RAISE (if not max raises)
        """
        actions = []
        
        if self.game_state.current_player != player_id:
            raise ValueError(f"Not player {player_id}'s turn")
        
        opponent_id = 1 - player_id
        
        # Get current street's bets
        my_bets = self.game_state.public_bets[player_id]
        opp_bets = self.game_state.public_bets[opponent_id]
        
        # Compare how much each has bet this street
        my_street_contribution = sum(amount for _, amount in my_bets)
        opp_street_contribution = sum(amount for _, amount in opp_bets)
        
        # Can we raise? (not max raises yet and stacks allow)
        num_raises = sum(1 for action, _ in opp_bets if action == Action.RAISE)
        can_raise = num_raises < self.max_raises and self.game_state.stacks[player_id] > 0
        
        if my_street_contribution < opp_street_contribution:
            # Someone bet more than us
            actions.append(Action.FOLD)
            actions.append(Action.CALL)  # Match their bet
            if can_raise:
                actions.append(Action.RAISE)
        elif my_street_contribution == opp_street_contribution:
            # Amounts equal: can check or raise
            actions.append(Action.CHECK)
            if can_raise:
                actions.append(Action.RAISE)
        else:
            # We bet more (shouldn't happen in Kuhn, but for completeness)
            actions.append(Action.CALL)
            if can_raise:
                actions.append(Action.RAISE)
        
        return actions
    
    def step(self, player_id: int, action: Action, amount: Optional[int] = None) -> Tuple[GameState, ObservableState, bool]:
        """
        Execute action and return new state.
        
        Args:
            player_id: Actor
            action: Action to take
            amount: Bet amount (calculated automatically for CALL if None)
            
        Returns:
            (full_state, observable_state, is_terminal)
        """
        if self.game_state.is_terminal:
            raise ValueError("Game is terminal")
        
        if self.game_state.current_player != player_id:
            raise ValueError(f"Not player {player_id}'s turn")
        
        # Calculate betting amounts based on action type
        if action == Action.CHECK:
            amount = 0
        elif action == Action.CALL:
            # Calculate call amount to match opponent's contribution
            opponent_id = 1 - player_id
            my_street_contribution = sum(amt for _, amt in self.game_state.public_bets[player_id])
            opp_street_contribution = sum(amt for _, amt in self.game_state.public_bets[opponent_id])
            amount = opp_street_contribution - my_street_contribution
            # Ensure amount is at least the minimum (shouldn't be negative)
            amount = max(0, amount)
        elif action == Action.RAISE:
            # For RAISE, if amount is provided, use it directly
            # If amount is None, calculate default raise (call + 1 chip)
            if amount is None:
                opponent_id = 1 - player_id
                my_street_contribution = sum(amt for _, amt in self.game_state.public_bets[player_id])
                opp_street_contribution = sum(amt for _, amt in self.game_state.public_bets[opponent_id])
                call_amount = max(0, opp_street_contribution - my_street_contribution)
                amount = call_amount + 1  # Raise by 1 chip over the call
        else:
            amount = 0
        
        # Record action
        self.game_state.public_bets[player_id].append((action, amount))
        
        # Update pot and stacks (only for CALL and RAISE)
        if action in [Action.CALL, Action.RAISE]:
            self.game_state.stacks[player_id] -= amount
            self.game_state.pot += amount
        
        # Check terminal conditions
        if action == Action.FOLD:
            # Other player wins
            self.game_state.is_terminal = True
            other_id = 1 - player_id
            payoff = self.game_state.pot

            # Update stacks: winner gets the pot
            self.game_state.stacks[other_id] += payoff

            self.game_state.payoffs = [
                payoff if i == other_id else -payoff
                for i in range(self.NUM_PLAYERS)
            ]
        else:
            # Check if both players have acted and contributions are equal
            other_id = 1 - player_id
            my_contribution = sum(amt for _, amt in self.game_state.public_bets[player_id])
            other_contribution = sum(amt for _, amt in self.game_state.public_bets[other_id])
            
            # Game only terminates when both players have acted AND contributions are equal
            # AND the action that made them equal was CHECK or CALL
            both_acted = (len(self.game_state.public_bets[player_id]) > 0 and 
                         len(self.game_state.public_bets[other_id]) > 0)
            
            if both_acted and my_contribution == other_contribution and action in [Action.CHECK, Action.CALL]:
                # Game reaches showdown
                self.game_state.is_terminal = True
                my_card = self.game_state.private_cards[player_id]
                other_card = self.game_state.private_cards[other_id]

                if my_card > other_card:
                    payoff = self.game_state.pot
                elif other_card > my_card:
                    payoff = -self.game_state.pot
                else:
                    payoff = 0  # Tie (shouldn't happen in Kuhn)

                # Update stacks: winner gets the pot
                winner_id = player_id if payoff > 0 else (1 - player_id) if payoff < 0 else None
                if winner_id is not None:
                    self.game_state.stacks[winner_id] += self.game_state.pot

                self.game_state.payoffs = [payoff if i == player_id else -payoff
                                          for i in range(self.NUM_PLAYERS)]
            else:
                # Pass to next player
                self.game_state.current_player = 1 - player_id
        
        obs = self._get_observable_state(1 - player_id)
        return self.game_state, obs, self.game_state.is_terminal
    
def get_payoff(self, player_id: int) -> float:
        """Get terminal payoff for player (only valid if terminal)."""
        if not self.game_state.is_terminal:
            raise ValueError("Game not terminal")
        if self.game_state.payoffs is None:
            raise ValueError("Payoffs not set")
        return self.game_state.payoffs[player_id]


class LeduckPoker:
    """
    Simplified Leduc poker: 2 streets, 6-card deck.
    More complex than Kuhn, allows testing scaling.
    (TODO: Full implementation if needed)
    """
    pass
