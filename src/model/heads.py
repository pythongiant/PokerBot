"""
Learned latent transition model and value/policy heads for poker belief state.

Transition Model: z_{t+1} = g_θ(z_t, a_t)
  - Implicitly learns opponent mixed strategies
  - Learns to update belief when actions are observed
  - Deterministic for simplicity (can extend to probabilistic)

Value Head: V_θ(z_t) -> scalar
  - Estimates expected cumulative reward from belief state
  - Trained against bootstrapped targets from rollouts
  - Represents counterfactual EV

Policy Head: π_θ(a | z_t) -> [0,1]^|A|
  - Outputs action logits
  - Masked to legal actions only
  - Trained via cross-entropy against improved targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class LatentTransitionModel(nn.Module):
    """
    Learn latent dynamics: z_{t+1} = g_θ(z_t, a_t)
    
    This model learns to update the belief state when an action is observed.
    The implicit assumption is that the opponent's strategy is encoded in
    how they react to various game states.
    
    We use a simple MLP: [z_t; one_hot(a_t)] -> z_{t+1}
    """
    
    def __init__(self, latent_dim: int, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Simple MLPTransition
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Update belief state given action.
        
        Args:
            z: (batch, latent_dim) - current belief state
            action: (batch,) - action indices (0, 1, 2, 3 for FOLD, CALL, RAISE, CHECK)
            
        Returns:
            z_next: (batch, latent_dim) - updated belief state
        """
        batch_size = z.size(0)
        device = z.device
        
        # One-hot encode action
        action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        
        # Concatenate and pass through network
        x = torch.cat([z, action_one_hot], dim=-1)
        z_next = self.net(x)
        
        return z_next


class ValueHead(nn.Module):
    """
    Value function head: V_θ(z) -> scalar chip EV
    
    Predicts the expected value of reaching showdown from the current belief state.
    This is crucial for credit assignment and search-based training.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of belief state.
        
        Args:
            z: (batch, latent_dim)
            
        Returns:
            v: (batch, 1) - scalar value per sample
        """
        return self.net(z)


class PolicyHead(nn.Module):
    """
    Policy head: π_θ(a | z_t) -> logits over actions
    
    Outputs logits for all possible actions.
    Masking of illegal actions happens during sampling/inference.
    """
    
    def __init__(self, latent_dim: int, num_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        
        self.num_actions = num_actions
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits.
        
        Args:
            z: (batch, latent_dim)
            
        Returns:
            logits: (batch, num_actions)
        """
        return self.net(z)
    
    def sample_action(self, z: torch.Tensor, legal_actions: List[int],
                      temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy, respecting legal action mask.
        
        Args:
            z: (batch, latent_dim) or (latent_dim,) for single state
            legal_actions: List[int] or List[List[int]] - legal actions per sample
            temperature: Softmax temperature (>1 = more entropy)
            
        Returns:
            action: (batch,) - sampled action indices
            log_prob: (batch,) - log probability of sampled action
        """
        logits = self.forward(z)
        
        # Handle single state
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            single_state = True
        else:
            single_state = False
        
        # Apply temperature
        logits = logits / temperature
        
        # Mask illegal actions
        batch_size = logits.size(0)
        mask = torch.full_like(logits, float('-inf'))
        
        # If single legal_actions list, apply to all samples
        if isinstance(legal_actions[0], int):
            for action_idx in legal_actions:
                mask[:, action_idx] = 0.0
        else:
            # Per-sample legal actions
            for i, actions in enumerate(legal_actions):
                for action_idx in actions:
                    mask[i, action_idx] = 0.0
        
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        
        # Sample
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if single_state:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        
        return action, log_prob
    
    def get_action_probabilities(self, z: torch.Tensor, legal_actions: Optional[List[int]] = None,
                                 temperature: float = 1.0) -> torch.Tensor:
        """
        Get action probability distribution, optionally masked.
        
        Args:
            z: (batch, latent_dim)
            legal_actions: Optional list of legal actions
            temperature: Softmax temperature
            
        Returns:
            probs: (batch, num_actions) - probability distribution
        """
        logits = self.forward(z) / temperature
        
        if legal_actions is not None:
            mask = torch.full_like(logits, float('-inf'))
            for action_idx in legal_actions:
                mask[:, action_idx] = 0.0
            logits = logits + mask
        
        return F.softmax(logits, dim=-1)


class OpponentRangePredictor(nn.Module):
    """
    Optional auxiliary head: Predict opponent's card distribution.
    
    This is useful for:
    - Interpretability: Can we extract opponent ranges from attention?
    - Regularization: Encouraging belief state to encode card knowledge
    - Analysis: Seeing if model learns exploitable patterns
    """
    
    def __init__(self, latent_dim: int, num_cards: int = 3, hidden_dim: int = 128):
        super().__init__()
        
        self.num_cards = num_cards
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_cards),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict opponent's card distribution.
        
        Args:
            z: (batch, latent_dim)
            
        Returns:
            logits: (batch, num_cards) - logits for each possible card
        """
        logits = self.net(z)
        return logits
    
    def get_card_probabilities(self, z: torch.Tensor, available_cards: List[int]) -> torch.Tensor:
        """
        Get card probabilities masked to available cards.
        
        Args:
            z: (batch, latent_dim)
            available_cards: List[int] - cards that haven't been dealt
            
        Returns:
            probs: (batch, num_cards)
        """
        logits = self.forward(z)
        
        # Mask unavailable cards
        mask = torch.full_like(logits, float('-inf'))
        for card_idx in available_cards:
            mask[:, card_idx] = 0.0
        
        logits = logits + mask
        return F.softmax(logits, dim=-1)
