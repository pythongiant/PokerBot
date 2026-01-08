"""
Unified Poker Transformer Agent Model.

Combines:
- BeliefStateTransformer: z_t = f_θ(o_{1:t})
- LatentTransitionModel: z_{t+1} = g_θ(z_t, a_t)
- ValueHead: V_θ(z_t)
- PolicyHead: π_θ(a | z_t)
- [Optional] OpponentRangePredictor

The full model is trained end-to-end via self-play with targets from search/rollouts.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional

from .transformer import BeliefStateTransformer
from .heads import (LatentTransitionModel, ValueHead, PolicyHead, 
                    OpponentRangePredictor)


class PokerTransformerAgent(nn.Module):
    """
    Complete Poker AI Agent: Belief State + Dynamics + Value + Policy
    
    Architecture Overview:
    ┌─ Observable History     (cards, betting)
    │
    ├─> BeliefStateTransformer (causal attention)
    │                    ↓
    │                  z_t (latent belief)
    │                    ├──> LatentTransitionModel (z_t, a_t) → z_{t+1}
    │                    ├──> ValueHead → V_θ(z_t)
    │                    ├──> PolicyHead → π_θ(a | z_t)
    │                    └──> OpponentRangePredictor → P(opponent_card | z_t) [optional]
    │
    └─> [Search/Rollout in latent space]
                    ↓
        [Policy target, Value target]
                    ↓
        [Loss: KL(π*||π) + (V* - V)² + Transition error]
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        model_cfg = config.model
        
        # Belief encoder
        self.belief_encoder = BeliefStateTransformer(
            latent_dim=model_cfg.latent_dim,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
            ff_dim=model_cfg.ff_dim,
            max_seq_len=model_cfg.max_sequence_length,
            dropout=model_cfg.dropout,
            card_embed_dim=model_cfg.card_embed_dim,
            action_embed_dim=model_cfg.action_embed_dim,
            bet_embed_dim=model_cfg.bet_embed_dim,
            num_cards=3,  # Kuhn poker: 3 cards
            num_actions=4,  # FOLD, CALL, RAISE, CHECK
        )
        
        # Latent transition dynamics
        self.transition_model = LatentTransitionModel(
            latent_dim=model_cfg.latent_dim,
            action_dim=4,
            hidden_dim=model_cfg.policy_hidden_dim,
        )
        
        # Value and policy heads
        self.value_head = ValueHead(
            latent_dim=model_cfg.latent_dim,
            hidden_dim=model_cfg.value_hidden_dim,
        )
        
        self.policy_head = PolicyHead(
            latent_dim=model_cfg.latent_dim,
            num_actions=4,
            hidden_dim=model_cfg.policy_hidden_dim,
        )
        
        # Optional: Opponent range prediction
        self.opponent_range_predictor = None
        if model_cfg.predict_opponent_range:
            self.opponent_range_predictor = OpponentRangePredictor(
                latent_dim=model_cfg.latent_dim,
                num_cards=3,
                hidden_dim=model_cfg.policy_hidden_dim,
            )
    
    def encode_belief(self, observations: List) -> Tuple[torch.Tensor, Dict]:
        """
        Encode observable game history into latent belief state.
        
        Args:
            observations: List of ObservableState objects
            
        Returns:
            belief_states: (batch, latent_dim)
            attention_info: Dict with attention weights for analysis
        """
        belief_states, attention_info = self.belief_encoder(observations)
        return belief_states, attention_info
    
    def predict_next_belief(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Predict next belief state given actions.
        
        Args:
            z: (batch, latent_dim) - current belief
            actions: (batch,) - action indices taken
            
        Returns:
            z_next: (batch, latent_dim) - next belief
        """
        return self.transition_model(z, actions)
    
    def predict_value(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict value (expected payoff) from belief state.
        
        Args:
            z: (batch, latent_dim)
            
        Returns:
            v: (batch, 1)
        """
        return self.value_head(z)
    
    def predict_policy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict action logits from belief state.
        
        Args:
            z: (batch, latent_dim)
            
        Returns:
            logits: (batch, 4) - logits for FOLD, CALL, RAISE, CHECK
        """
        return self.policy_head(z)
    
    def predict_opponent_range(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Optionally predict opponent's card distribution.
        
        Args:
            z: (batch, latent_dim)
            
        Returns:
            logits: (batch, 3) or None
        """
        if self.opponent_range_predictor is not None:
            return self.opponent_range_predictor(z)
        return None
    
    def forward(self, observations: List) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode observations and predict all heads.
        
        Args:
            observations: List of ObservableState objects
            
        Returns:
            outputs: Dict with:
                - belief_states: (batch, latent_dim)
                - values: (batch, 1)
                - policy_logits: (batch, 4)
                - opponent_range_logits: (batch, 3) [optional]
                - attention_weights: For interpretability
        """
        # Encode beliefs
        belief_states, attention_info = self.encode_belief(observations)
        
        # Predict heads
        values = self.predict_value(belief_states)
        policy_logits = self.predict_policy(belief_states)
        
        # Opponent range [optional]
        opponent_logits = self.predict_opponent_range(belief_states)
        
        return {
            'belief_states': belief_states,
            'values': values,
            'policy_logits': policy_logits,
            'opponent_range_logits': opponent_logits,
            'attention_weights': attention_info['attention_weights'],
            'final_hidden': attention_info['final_hidden'],
        }
    
    def get_parameters(self, recurse: bool = True):
        """Get model parameters for optimization."""
        return self.parameters() if recurse else self.parameters()


class BeliefStateGeometry:
    """
    Analysis utilities for belief state geometry and interpretability.
    
    Key questions:
    - How is card information distributed across latent dimensions?
    - Can we extract opponent ranges from attention patterns?
    - What do transitions look like geometrically?
    """
    
    def __init__(self, agent: PokerTransformerAgent):
        self.agent = agent
    
    def get_attention_to_opponent_actions(self,
                                          attention_weights: List[torch.Tensor],
                                          action_history: List) -> torch.Tensor:
        """
        Extract attention weights focused on opponent's actions.

        Intuition: If the model learns well, attention to opponent actions
        should encode how likely different cards are.

        Args:
            attention_weights: From forward pass
            action_history: List of (player, action, amount) tuples

        Returns:
            opponent_attention: (batch, seq_len) - attention to opponent moves
        """
        # Find indices of opponent actions in sequence
        opponent_indices = []
        for i, (player, action, amount) in enumerate(action_history):
            if player == 1:  # Opponent
                opponent_indices.append(i + 1)  # +1 because we prepend own card

        if not opponent_indices:
            return None

        # Average attention across heads and layers
        # attention_weights is list of (batch, heads, seq_len, seq_len)
        avg_attention = torch.stack(attention_weights).mean(dim=(1, 2))  # (batch, seq_len, seq_len)

        # Get final position's attention to opponent actions
        final_attn = avg_attention[:, -1, :]  # (batch, seq_len)

        opponent_attention = final_attn[:, opponent_indices].mean(dim=-1)
        return opponent_attention

    def analyze_attention_semantics(self, attention_weights: List[torch.Tensor],
                                   action_history: List) -> Dict:
        """
        Analyze attention patterns with semantic meaning.

        Returns correlations between attention and game semantics.
        """
        results = {}

        # Attention to different action types
        action_types = {'FOLD': [], 'CALL': [], 'RAISE': [], 'CHECK': []}

        for layer_idx, attn_w in enumerate(attention_weights):
            # attn_w: (batch, heads, seq_len, seq_len)
            final_attn = attn_w[0, :, -1, :].mean(dim=0)  # Average over heads

            # Correlate with action types in history
            for pos, (player, action, amount) in enumerate(action_history):
                if pos + 1 < len(final_attn):  # +1 for card position
                    attention_weight = final_attn[pos + 1].item()
                    action_types[action.name].append(attention_weight)

            results[f'layer_{layer_idx}'] = {
                'attention_by_action': {
                    action: np.mean(weights) if weights else 0.0
                    for action, weights in action_types.items()
                }
            }

        return results
    
    def belief_state_variance(self, belief_states: torch.Tensor) -> torch.Tensor:
        """
        Compute variance across belief dimensions.
        
        High variance = information is spread across dimensions
        Low variance = information is concentrated
        
        Args:
            belief_states: (batch, latent_dim)
            
        Returns:
            variance: (latent_dim,)
        """
        return belief_states.var(dim=0)
    
    def value_landscape(self, z_samples: torch.Tensor) -> torch.Tensor:
        """
        Sample belief states and evaluate value function.
        
        Useful for understanding value function geometry.
        
        Args:
            z_samples: (batch, latent_dim)
            
        Returns:
            values: (batch, 1)
        """
        return self.agent.predict_value(z_samples)
    
    def attention_flow_analysis(self, attention_weights: List[torch.Tensor]) -> Dict:
        """
        Analyze how attention flows through the network.
        
        Returns statistics about what positions attend to what over layers.
        """
        results = {}
        
        for layer_idx, attn_w in enumerate(attention_weights):
            # attn_w: (batch, num_heads, seq_len, seq_len)
            batch_size, num_heads, seq_len, _ = attn_w.shape
            
            # Average over heads
            avg_attn = attn_w.mean(dim=1)  # (batch, seq_len, seq_len)
            
            # Final token attention distribution
            final_attn = avg_attn[:, -1, :].mean(dim=0)  # (seq_len,)
            
            results[f'layer_{layer_idx}'] = {
                'final_token_attention': final_attn,
                'entropy': -(final_attn * (final_attn + 1e-10).log()).sum(),
            }
        
        return results
