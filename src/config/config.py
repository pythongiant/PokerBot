"""
Configuration and hyperparameters for Poker Transformer Agent.

This module centralizes all experimental settings to ensure reproducibility
and easy ablation studies.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class EnvironmentConfig:
    """Configuration for poker game environment."""
    
    # Game type: "kuhn" or "leduc"
    game_type: str = "kuhn"
    
    # Initial stack size (chips) for each player
    initial_stack: int = 100
    
    # Ante / blind structure
    ante: int = 1
    
    # Maximum number of raises per street (to limit game tree)
    max_raises: int = 4
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for Transformer belief model."""
    
    # Latent belief state dimension
    latent_dim: int = 64
    
    # Transformer architecture
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256  # Feed-forward hidden dimension
    dropout: float = 0.1
    
    # Embedding dimensions
    card_embed_dim: int = 16
    action_embed_dim: int = 16
    bet_embed_dim: int = 16
    
    # Maximum sequence length (for positional encoding)
    max_sequence_length: int = 128
    
    # Transition model type: "deterministic" or "probabilistic"
    transition_type: str = "deterministic"
    
    # Value head configuration
    value_hidden_dim: int = 128
    
    # Policy head configuration
    policy_hidden_dim: int = 128
    
    # Enable opponent range prediction head (optional)
    predict_opponent_range: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    
    # Number of self-play iterations
    num_iterations: int = 1000
    
    # Games per iteration
    games_per_iteration: int = 128
    
    # Batch size for training
    batch_size: int = 32
    
    # Learning rate
    learning_rate: float = 1e-3
    
    # Optimizer: "adam" or "sgd"
    optimizer: str = "adam"
    
    # Weight decay for regularization
    weight_decay: float = 1e-5
    
    # Gradient clipping norm
    grad_clip: float = 1.0
    
    # Loss weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "policy": 1.0,
        "value": 1.0,
        "transition": 0.5,
        "opponent_range": 0.1,
    })
    
    # Search/rollout configuration
    search_type: str = "mcts"  # "mcts" or "rollout"
    num_simulations: int = 50
    rollout_depth: int = 10
    
    # Bootstrap value targets with mixing
    value_bootstrap_mix: float = 0.99  # λ for bootstrap
    
    # Use mixed precision training
    use_amp: bool = False
    
    # Checkpoint frequency (iterations)
    checkpoint_freq: int = 10
    
    # Early stopping patience
    early_stopping_patience: int = 50


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    # Games for head-to-head evaluation
    eval_games: int = 100
    
    # Compute exploitability (requires Nash solver)
    compute_exploitability: bool = False
    
    # Probe belief state via attention analysis
    probe_beliefs: bool = True
    
    # Random baseline for comparison
    eval_vs_random: bool = True
    
    # Self-play evaluation over time
    eval_vs_checkpoint: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    
    # Experiment name for logging
    name: str = "poker_transformer_default"
    
    # Logging directory
    log_dir: str = "./logs"
    
    # Device: "cuda" or "cpu"
    device: str = "cpu"
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Sub-configurations
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict for logging."""
        return {
            "name": self.name,
            "environment": self.environment.__dict__,
            "model": self.model.__dict__,
            "training": {k: v for k, v in self.training.__dict__.items() 
                        if k != "loss_weights"},
            "evaluation": self.evaluation.__dict__,
        }


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()
