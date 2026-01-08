"""
Self-play game generation and latent space search.

This module handles:
1. Running self-play games using the current agent
2. MCTS or rollout-based search in latent space
3. Generating training targets (policy & value) from search results
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import deque
import random

from src.environment import KuhnPoker, Action, ObservableState


class SelfPlayGame:
    """One complete self-play game with full history for training."""
    
    def __init__(self):
        self.observations: List[ObservableState] = []  # What each player sees
        self.actions: List[int] = []  # Action taken at each step
        self.rewards: Dict[int, float] = {}  # Final payoffs
        self.search_policies: List[np.ndarray] = []  # π* from search
        self.search_values: List[float] = []  # V* from rollouts


class LatentSpaceSearcher:
    """
    MCTS-style search in latent space.
    
    At each node, we:
    1. Use the learned transition model to predict next latent state
    2. Use the value head for rollout evaluation
    3. Use the policy head for action selection
    """
    
    def __init__(self, agent, env: KuhnPoker, num_simulations: int = 50, 
                 rollout_depth: int = 10, temperature: float = 1.0):
        """
        Args:
            agent: PokerTransformerAgent
            env: Game environment
            num_simulations: MCTS simulations per position
            rollout_depth: Rollout depth for value bootstrap
            temperature: Softmax temperature for search policy
        """
        self.agent = agent
        self.env = env
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.temperature = temperature
        self.device = next(agent.parameters()).device
    
    def search(self, game_state, player_id: int) -> Tuple[np.ndarray, float]:
        """
        Perform MCTS in latent space to improve policy and value.
        
        Args:
            game_state: Current KuhnPoker game state
            player_id: Current player
            
        Returns:
            search_policy: (num_actions,) - improved policy distribution
            search_value: scalar - estimated value of position
        """
        # Get current observation
        obs = self._get_obs(game_state, player_id)
        
        # Encode into belief state
        with torch.no_grad():
            belief, _ = self.agent.encode_belief([obs])
            belief = belief[0]  # (latent_dim,)
        
        # Initialize visit counts and value estimates
        num_actions = 4
        visit_counts = np.zeros(num_actions)
        value_sums = np.zeros(num_actions)
        
        legal_actions = game_state.current_player == player_id  # Simplified
        legal_actions = self._get_legal_actions(game_state)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Simulate: pick action, rollout, backup value
            value = self._simulate(game_state, player_id, belief, legal_actions)
            
            # Update stats for each legal action (uniform exploration for now)
            if legal_actions:
                action = np.random.choice(legal_actions)
                visit_counts[action] += 1
                value_sums[action] += value
        
        # Compute improved policy: proportional to visits
        search_policy = visit_counts / (visit_counts.sum() + 1e-8)
        
        # Compute value: average of simulations
        search_value = value_sums.sum() / (self.num_simulations + 1e-8)
        
        return search_policy, search_value
    
    def _simulate(self, game_state, player_id: int, belief: torch.Tensor,
                  legal_actions: List[int]) -> float:
        """
        One MCTS simulation: rollout from current latent state.
        
        Args:
            game_state: KuhnPoker state
            player_id: Whose value we're computing
            belief: (latent_dim,) current belief
            legal_actions: List of legal action indices
            
        Returns:
            rollout_value: Estimated value from rollout
        """
        current_belief = belief.clone().detach()
        current_state = game_state  # Would need to copy state for real impl
        depth = 0
        
        # Random rollout
        while depth < self.rollout_depth and not current_state.is_terminal:
            # Sample action from policy
            with torch.no_grad():
                policy_logits = self.agent.predict_policy(current_belief.unsqueeze(0))
                policy = F.softmax(policy_logits[0], dim=-1).cpu().numpy()
            
            # Mask to legal actions
            legal_mask = np.zeros(4)
            legal_mask[legal_actions] = 1.0
            policy = policy * legal_mask
            policy = policy / (policy.sum() + 1e-8)
            
            action = np.random.choice(4, p=policy)
            
            # Update belief via transition model
            with torch.no_grad():
                action_tensor = torch.tensor(action, device=self.device).unsqueeze(0)
                current_belief = self.agent.predict_next_belief(
                    current_belief.unsqueeze(0), action_tensor
                )[0]
            
            depth += 1
            # In real impl would step the environment too
        
        # Value from terminal or bootstrap
        if current_state.is_terminal:
            return float(current_state.payoffs[player_id])
        else:
            with torch.no_grad():
                value = self.agent.predict_value(current_belief.unsqueeze(0))[0, 0]
            return float(value)
    
    def _get_obs(self, game_state, player_id: int) -> ObservableState:
        """Convert KuhnPoker state to observable state."""
        from src.environment import ObservableState, Action as EnvAction
        
        action_history = []
        for p_id, actions in enumerate(game_state.public_bets):
            for action, amount in actions:
                action_history.append((p_id, action, amount))
        
        return ObservableState(
            public_cards=[],
            own_card=game_state.private_cards[player_id],
            action_history=action_history,
            current_player=game_state.current_player,
            stacks=game_state.stacks.copy(),
            pot=game_state.pot,
            street=game_state.street,
        )
    
    def _get_legal_actions(self, game_state) -> List[int]:
        """Get legal action indices."""
        if game_state.current_player == 0:
            # Simplified: both players have same legal set in Kuhn
            return [0, 1, 2, 3]  # All actions for now
        else:
            return [0, 1, 2, 3]


class SelfPlayBuffer:
    """
    Experience replay buffer for storing self-play games.
    
    Each game becomes a sequence of training examples:
    (observation, search_policy, search_value)
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.games: deque = deque(maxlen=max_size)
        self.device = None
    
    def add_game(self, game: SelfPlayGame):
        """Add completed self-play game."""
        self.games.append(game)
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample random games and create training batch.
        
        Returns:
            batch: Dict with observations, targets, etc.
        """
        if len(self.games) == 0:
            raise ValueError("Buffer is empty")
        
        # Sample games
        sampled_games = random.sample(list(self.games), 
                                     min(batch_size, len(self.games)))
        
        observations = []
        policy_targets = []
        value_targets = []
        
        for game in sampled_games:
            # Each game contributes one (or more) examples
            # For simplicity: use last observation + targets
            if game.observations:
                observations.append(game.observations[-1])
                if game.search_policies:
                    policy_targets.append(game.search_policies[-1])
                if game.search_values:
                    value_targets.append(game.search_values[-1])
        
        return {
            'observations': observations,
            'policy_targets': np.stack(policy_targets) if policy_targets else None,
            'value_targets': np.array(value_targets) if value_targets else None,
        }
    
    def get_size(self) -> int:
        return len(self.games)


def run_self_play_game(agent, env: KuhnPoker, searcher: Optional[LatentSpaceSearcher] = None) -> SelfPlayGame:
    """
    Run one complete self-play game.
    
    Args:
        agent: PokerTransformerAgent
        env: Game environment
        searcher: Optional search engine for targets
        
    Returns:
        game: SelfPlayGame with history
    """
    game = SelfPlayGame()
    
    game_state, obs = env.reset()
    current_player = 0
    
    while not game_state.is_terminal:
        game.observations.append(obs)
        
        # Get action from agent
        with torch.no_grad():
            device = next(agent.parameters()).device
            belief, _ = agent.encode_belief([obs])
            policy_logits = agent.predict_policy(belief)
            
            # Get legal actions
            legal_actions = env.get_legal_actions(current_player)
            
            # Sample action (with some exploration)
            if np.random.random() < 0.1:  # 10% random
                action = np.random.choice(legal_actions)
            else:
                policy = F.softmax(policy_logits[0], dim=-1).cpu().numpy()
                legal_mask = np.zeros(4)
                legal_mask[legal_actions] = 1.0
                policy = policy * legal_mask
                policy = policy / (policy.sum() + 1e-8)
                action = np.random.choice(4, p=policy)
        
        game.actions.append(action)
        
        # Optionally get search targets
        if searcher is not None:
            search_policy, search_value = searcher.search(game_state, current_player)
            game.search_policies.append(search_policy)
            game.search_values.append(search_value)
        
        # Step environment
        game_state, obs, _ = env.step(current_player, Action(action), amount=1)
        current_player = 1 - current_player
    
    # Record final payoffs
    game.rewards[0] = game_state.payoffs[0]
    game.rewards[1] = game_state.payoffs[1]
    
    return game
