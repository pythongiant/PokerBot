"""
Transformer-based belief state encoder for poker.

Core insight: The Transformer acts as a belief state aggregator,
learning to encode the entire observable game history (cards, betting)
into a latent representation z_t that captures:
- Own card strength
- Opponent's likely range (through learned attention patterns)
- Betting dynamics and game progress

This is causal (masked attention) to respect temporal ordering.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Absolute positional encoding for sequence positions."""
    
    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len].to(x.device)
        x = x + pe
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking (no looking into future).
    
    Why causal? Betting decisions at time t shouldn't see future actions.
    Opponent strategies must be inferred from past, not future.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len) - for analysis
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = rearrange_qkv(qkv, batch_size, seq_len, self.num_heads)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create causal mask (lower triangular: each position can only see itself and past)
        if mask is None:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
            mask = causal_mask.bool()
        
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)
        
        return output, attn_weights


def rearrange_qkv(qkv: torch.Tensor, batch_size: int, seq_len: int, 
                   num_heads: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape QKV from (B, L, 3D) to separate (B, H, L, D) for each."""
    d = qkv.size(-1) // 3
    head_dim = d // num_heads
    
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
    q, k, v = qkv[0], qkv[1], qkv[2]
    return q, k, v


class TransformerBlock(nn.Module):
    """Single Transformer block: attention + FFN with residual connections."""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network: d_model -> ff_dim -> d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional causal mask
            
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: From attention layer
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attn(x, mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


class BeliefStateTransformer(nn.Module):
    """
    Causal Transformer encoder: z_t = f_θ(o_{1:t})
    
    Encodes observable history (cards + betting) into latent belief state.
    
    Key design:
    - Causal attention: respects temporal ordering
    - Separate embeddings for: cards, actions, amounts
    - Variable-length sequences via attention masking
    """
    
    def __init__(self, latent_dim: int, num_heads: int, num_layers: int, 
                 ff_dim: int, max_seq_len: int = 128, dropout: float = 0.1,
                 card_embed_dim: int = 16, action_embed_dim: int = 16,
                 bet_embed_dim: int = 16, num_cards: int = 3, num_actions: int = 4):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Embeddings
        self.card_embed = nn.Embedding(num_cards + 1, card_embed_dim)  # +1 for padding
        self.action_embed = nn.Embedding(num_actions + 1, action_embed_dim)  # +1 for padding
        self.bet_embed = nn.Linear(1, bet_embed_dim)  # Continuous bet amount
        
        # Input projection: concatenate all embeddings and project to latent_dim
        input_size = card_embed_dim + action_embed_dim + bet_embed_dim
        self.input_proj = nn.Linear(input_size, latent_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_len, latent_dim, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(latent_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def encode_observation(self, obs) -> torch.Tensor:
        """
        Encode observable state into sequence of embeddings.
        
        Observation contains:
        - own_card: int (0-2 in Kuhn)
        - action_history: list of (player, action, amount)
        
        Returns:
            sequence: (seq_len, input_size) - concatenated embeddings
        """
        sequence = []
        
        # Start with own card
        card_embed = self.card_embed(torch.tensor(obs.own_card))
        action_embed = self.action_embed(torch.tensor(4))  # Padding, means "no action yet"
        bet_embed = self.bet_embed(torch.tensor([[0.0]]))
        
        token = torch.cat([card_embed, action_embed, bet_embed.squeeze(0)], dim=-1)
        sequence.append(token)
        
        # Add action history
        for player_id, action, amount in obs.action_history:
            action_val = action.value  # Enum to int
            card_embed = self.card_embed(torch.tensor(obs.own_card))
            action_embed = self.action_embed(torch.tensor(action_val))
            bet_embed = self.bet_embed(torch.tensor([[float(amount)]]))
            
            token = torch.cat([card_embed, action_embed, bet_embed.squeeze(0)], dim=-1)
            sequence.append(token)
        
        if sequence:
            return torch.stack(sequence, dim=0)  # (seq_len, input_size)
        else:
            # Empty history: just own card
            card_embed = self.card_embed(torch.tensor(obs.own_card))
            action_embed = self.action_embed(torch.tensor(4))
            bet_embed = self.bet_embed(torch.tensor([[0.0]]))
            token = torch.cat([card_embed, action_embed, bet_embed.squeeze(0)], dim=-1)
            return token.unsqueeze(0)  # (1, input_size)
    
    def forward(self, obs_list) -> Tuple[torch.Tensor, dict]:
        """
        Process batch of observations.
        
        Args:
            obs_list: List of ObservableState objects
            
        Returns:
            belief_states: (batch, latent_dim) - final latent state per game
            attention_info: dict with attention weights for analysis
        """
        batch_size = len(obs_list)
        device = next(self.parameters()).device
        
        # Pad sequences to same length
        sequences = [self.encode_observation(obs) for obs in obs_list]
        max_len = max(seq.size(0) for seq in sequences)
        
        # Create padded batch
        padded_batch = torch.zeros(batch_size, max_len, sequences[0].size(-1), device=device)
        padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        
        for i, seq in enumerate(sequences):
            seq_len = seq.size(0)
            padded_batch[i, :seq_len] = seq.to(device)
            padding_mask[i, seq_len:] = True
        
        # Project to latent dimension
        x = self.input_proj(padded_batch)
        x = self.pos_encoding(x)
        
        # Process through transformer layers
        attn_weights_list = []
        for layer in self.transformer_layers:
            x, attn_w = layer(x, mask=padding_mask)
            attn_weights_list.append(attn_w)
        
        # Extract belief state: last non-padded token
        belief_states = []
        for i in range(batch_size):
            seq_len = max_len - padding_mask[i].sum().item()
            belief_states.append(x[i, seq_len - 1])  # Last actual token
        
        belief_states = torch.stack(belief_states, dim=0)
        
        attention_info = {
            'attention_weights': attn_weights_list,
            'final_hidden': x,
        }
        
        return belief_states, attention_info
