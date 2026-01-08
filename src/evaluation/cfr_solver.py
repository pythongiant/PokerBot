"""
Counterfactual Regret Minimization (CFR) solver for Kuhn poker.

Used to compute Nash equilibrium strategies and exploitability metrics.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import logging

from src.environment import KuhnPoker, Action


class KuhnCFRSolver:
    """
    CFR solver for Kuhn poker.

    Computes exact Nash equilibrium using regret matching.
    """

    def __init__(self, max_iterations: int = 10000):
        self.max_iterations = max_iterations

        # Game info
        self.NUM_CARDS = 3  # J, Q, K
        self.NUM_ACTIONS = 2  # PASS, BET (simplified from 4 actions)

        # Regret and strategy storage
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        self.node_util = {}

        # Precompute all infosets (card, history)
        self._build_infosets()

        self.logger = logging.getLogger(__name__)

    def _build_infosets(self):
        """Build all possible information sets."""
        self.infosets = []

        # Kuhn poker infosets: (card, history_string)
        cards = ['J', 'Q', 'K']
        actions = ['p', 'b']  # pass, bet

        # All possible histories
        histories = [
            "",      # No actions
            "p",     # Player 1 passed
            "b",     # Player 1 bet
            "pp",    # Both passed
            "pb",    # P1 passed, P2 bet
            "bp",    # P1 bet, P2 passed
            "bb",    # Both bet
        ]

        for card in cards:
            for history in histories:
                if self._is_valid_infoset(card, history):
                    self.infosets.append((card, history))

    def _is_valid_infoset(self, card: str, history: str) -> bool:
        """Check if infoset is reachable."""
        # Basic validation - more complex games would need proper validation
        return len(history) <= 2  # Kuhn has at most 2 actions

    def _get_infoset_key(self, card: int, history: str) -> str:
        """Convert card int and history to infoset key."""
        card_names = ['J', 'Q', 'K']
        return f"{card_names[card]}_{history}"

    def train(self, iterations: Optional[int] = None) -> Dict:
        """
        Train CFR and return Nash equilibrium strategy.

        Returns:
            nash_strategy: Dict[infoset -> action_probs]
        """
        iterations = iterations or self.max_iterations

        for i in range(iterations):
            if i % 1000 == 0:
                self.logger.info(f"CFR iteration {i}/{iterations}")

            # Traverse game tree for both players
            for player in [0, 1]:
                self._cfr('', player, [None, None])

        # Average strategies
        nash_strategy = {}
        for infoset in self.infosets:
            key = self._get_infoset_key(infoset[0], infoset[1])
            if key in self.strategy_sum:
                total = sum(self.strategy_sum[(key, a)] for a in range(self.NUM_ACTIONS))
                if total > 0:
                    nash_strategy[key] = [
                        self.strategy_sum[(key, a)] / total for a in range(self.NUM_ACTIONS)
                    ]
                else:
                    nash_strategy[key] = [1.0 / self.NUM_ACTIONS] * self.NUM_ACTIONS

        return nash_strategy

    def _cfr(self, history: str, player: int, cards: List[Optional[int]]) -> float:
        """
        CFR traversal.

        Simplified Kuhn implementation.
        """
        # Check if terminal
        if self._is_terminal(history):
            return self._get_payoff(history, cards, player)

        # Get current player
        current_player = len(history) % 2

        if current_player == player:
            # Our turn - compute counterfactual regret
            infoset = self._get_infoset_key(cards[player], history)
            strategy = self._get_strategy(infoset)

            util = 0
            node_util = np.zeros(self.NUM_ACTIONS)

            for a in range(self.NUM_ACTIONS):
                # Try action
                new_history = history + ('p' if a == 0 else 'b')
                new_cards = cards.copy()

                # Recurse
                node_util[a] = self._cfr(new_history, player, new_cards)
                util += strategy[a] * node_util[a]

            # Update regrets
            for a in range(self.NUM_ACTIONS):
                regret = node_util[a] - util
                self.regret_sum[(infoset, a)] += regret

            return util
        else:
            # Opponent's turn
            infoset = self._get_infoset_key(cards[current_player], history)
            strategy = self._get_strategy(infoset)

            util = 0
            for a in range(self.NUM_ACTIONS):
                prob = strategy[a]
                new_history = history + ('p' if a == 0 else 'b')
                new_cards = cards.copy()

                util += prob * self._cfr(new_history, player, new_cards)

            return util

    def _get_strategy(self, infoset: str) -> np.ndarray:
        """Get current strategy for infoset using regret matching."""
        strategy = np.zeros(self.NUM_ACTIONS)

        # Positive regrets only
        for a in range(self.NUM_ACTIONS):
            strategy[a] = max(0, self.regret_sum[(infoset, a)])

        total = strategy.sum()
        if total > 0:
            strategy /= total
        else:
            strategy = np.ones(self.NUM_ACTIONS) / self.NUM_ACTIONS

        # Add to strategy sum for averaging
        for a in range(self.NUM_ACTIONS):
            self.strategy_sum[(infoset, a)] += strategy[a]

        return strategy

    def _is_terminal(self, history: str) -> bool:
        """Check if history is terminal."""
        if len(history) < 2:
            return False

        # Both players acted
        if len(history) == 2:
            # pp: both pass -> showdown
            # pb: P1 pass, P2 bet -> P2 wins
            # bp: P1 bet, P2 pass -> P1 wins
            # bb: both bet -> showdown
            return True

        return False

    def _get_payoff(self, history: str, cards: List[int], player: int) -> float:
        """Get payoff for terminal history."""
        if not self._is_terminal(history):
            return 0

        p1_card, p2_card = cards

        if history == "pp":  # Both pass
            if p1_card > p2_card:
                return 1 if player == 0 else -1
            else:
                return -1 if player == 0 else 1

        elif history == "pb":  # P1 pass, P2 bet
            return -1 if player == 0 else 1  # P2 wins bet

        elif history == "bp":  # P1 bet, P2 pass
            return 1 if player == 0 else -1  # P1 wins bet

        elif history == "bb":  # Both bet
            if p1_card > p2_card:
                return 2 if player == 0 else -2  # Higher bet
            else:
                return -2 if player == 0 else 2

        return 0

    def compute_exploitability(self, policy: Dict[str, List[float]]) -> float:
        """
        Compute exploitability of a policy vs Nash equilibrium.

        Returns best-response value (exploitability).
        """
        # For simplicity, compute expected value vs Nash for both players
        # and take the maximum (true exploitability)

        nash_strategy = self.train()

        def best_response_value(player: int) -> float:
            """Compute best response value for player vs Nash opponent."""
            total_value = 0
            count = 0

            # Enumerate all possible card assignments
            for p1_card in range(3):
                for p2_card in range(3):
                    if p1_card == p2_card:
                        continue

                    cards = [p1_card, p2_card] if player == 0 else [p2_card, p1_card]
                    value = self._best_response_game_value('', player, cards, nash_strategy)
                    total_value += value
                    count += 1

            return total_value / count if count > 0 else 0

        br_p0 = best_response_value(0)
        br_p1 = best_response_value(1)

        exploitability = max(abs(br_p0), abs(br_p1))
        return exploitability

    def _best_response_game_value(self, history: str, br_player: int,
                                cards: List[int], nash_strategy: Dict) -> float:
        """Compute best response value in subgame."""
        if self._is_terminal(history):
            return self._get_payoff(history, cards, br_player)

        current_player = len(history) % 2

        if current_player == br_player:
            # Best response player's turn - choose best action
            best_value = -float('inf')

            for action in range(self.NUM_ACTIONS):
                new_history = history + ('p' if action == 0 else 'b')
                value = self._best_response_game_value(new_history, br_player, cards, nash_strategy)
                best_value = max(best_value, value)

            return best_value
        else:
            # Nash opponent's turn
            infoset = self._get_infoset_key(cards[current_player], history)
            strategy = nash_strategy.get(infoset, [0.5, 0.5])  # Default uniform

            value = 0
            for action in range(self.NUM_ACTIONS):
                prob = strategy[action]
                new_history = history + ('p' if action == 0 else 'b')
                value += prob * self._best_response_game_value(new_history, br_player, cards, nash_strategy)

            return value