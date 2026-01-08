"""
Example usage patterns for Poker Transformer Agent.

Each example shows a complete workflow: training + evaluation.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import ExperimentConfig, ModelConfig, TrainingConfig
from src.training import PokerTrainer
from src.evaluation import PokerEvaluator


# ============================================================================
# Example 1: Basic training with default config
# ============================================================================

def example_basic_training():
    """Minimal example: train agent with default hyperparameters."""
    
    # Create default config
    config = ExperimentConfig(
        name="example_basic",
        device="cpu",  # Change to "cuda" for GPU
    )
    
    # Optional: customize some hyperparameters
    config.training.num_iterations = 50  # Quick training
    config.training.games_per_iteration = 64
    
    # Create trainer and run
    trainer = PokerTrainer(config)
    trainer.train()
    
    # Evaluate
    evaluator = PokerEvaluator(trainer.agent, config)
    results = evaluator.evaluate_vs_random(num_games=100)
    
    print(f"Final evaluation: {results}")


# ============================================================================
# Example 2: Larger model with longer training
# ============================================================================

def example_larger_model():
    """Train a more capable model with more iterations."""
    
    config = ExperimentConfig(
        name="example_large_model",
        device="cpu",
        model=ModelConfig(
            latent_dim=128,           # Larger latent space
            num_heads=8,              # More attention heads
            num_layers=6,             # Deeper transformer
            ff_dim=512,               # Wider FFN
        ),
        training=TrainingConfig(
            num_iterations=200,       # More training
            games_per_iteration=256,  # More games per iteration
            batch_size=64,            # Larger batches
            learning_rate=5e-4,       # Lower LR for stability
        ),
    )
    
    trainer = PokerTrainer(config)
    trainer.train()
    
    evaluator = PokerEvaluator(trainer.agent, config)
    evaluator.run_full_evaluation()


# ============================================================================
# Example 3: Ablation study - no transition model
# ============================================================================

def example_ablation_no_transition():
    """Study effect of removing transition model."""
    
    config = ExperimentConfig(
        name="example_ablation_no_transition",
        device="cpu",
    )
    
    # Set transition loss weight to zero
    config.training.loss_weights['transition'] = 0.0
    
    trainer = PokerTrainer(config)
    trainer.train()
    
    evaluator = PokerEvaluator(trainer.agent, config)
    results = evaluator.evaluate_vs_random(num_games=100)
    
    print(f"Without transition model: {results}")


# ============================================================================
# Example 4: Belief state analysis
# ============================================================================

def example_belief_analysis():
    """Train and analyze learned belief state geometry."""
    
    from src.model import BeliefStateGeometry
    
    config = ExperimentConfig(
        name="example_belief_analysis",
        device="cpu",
    )
    
    trainer = PokerTrainer(config)
    trainer.train()
    
    # Analyze belief geometry
    geometry = BeliefStateGeometry(trainer.agent)
    
    # Generate belief states from random games
    import torch
    import numpy as np
    from src.environment import KuhnPoker, Action
    
    env = KuhnPoker(seed=42)
    beliefs = []
    
    for _ in range(50):
        game_state, obs = env.reset()
        while not game_state.is_terminal:
            with torch.no_grad():
                belief, _ = trainer.agent.encode_belief([obs])
            beliefs.append(belief.cpu().numpy())
            
            legal_actions = env.get_legal_actions(game_state.current_player)
            action = np.random.choice(legal_actions)
            game_state, obs, _ = env.step(game_state.current_player, Action(action))
    
    beliefs = np.array(beliefs)
    
    print(f"Belief state statistics:")
    print(f"  Shape: {beliefs.shape}")
    print(f"  Mean variance across dimensions: {beliefs.var(axis=0).mean():.4f}")
    print(f"  Min/max values: [{beliefs.min():.4f}, {beliefs.max():.4f}]")


# ============================================================================
# Example 5: Comparing search types
# ============================================================================

def example_search_comparison():
    """Compare MCTS vs. rollout targets."""
    
    # Train with MCTS
    config_mcts = ExperimentConfig(
        name="example_with_mcts",
        device="cpu",
        training=TrainingConfig(
            search_type="mcts",
            num_simulations=50,
        ),
    )
    config_mcts.training.num_iterations = 50
    config_mcts.training.games_per_iteration = 64
    
    trainer_mcts = PokerTrainer(config_mcts)
    trainer_mcts.train()
    
    # Train with rollout
    config_rollout = ExperimentConfig(
        name="example_rollout_only",
        device="cpu",
        training=TrainingConfig(
            search_type="rollout",
            rollout_depth=10,
        ),
    )
    config_rollout.training.num_iterations = 50
    config_rollout.training.games_per_iteration = 64
    
    trainer_rollout = PokerTrainer(config_rollout)
    trainer_rollout.train()
    
    # Compare
    eval_mcts = PokerEvaluator(trainer_mcts.agent, config_mcts)
    eval_rollout = PokerEvaluator(trainer_rollout.agent, config_rollout)
    
    results_mcts = eval_mcts.evaluate_vs_random(num_games=100)
    results_rollout = eval_rollout.evaluate_vs_random(num_games=100)
    
    print(f"MCTS results: {results_mcts}")
    print(f"Rollout results: {results_rollout}")


# ============================================================================
# Example 6: Custom training loop with callbacks
# ============================================================================

def example_custom_training():
    """Advanced: custom training with monitoring."""
    
    config = ExperimentConfig(
        name="example_custom",
        device="cpu",
    )
    
    trainer = PokerTrainer(config)
    
    # Modify training loop (pseudo-code, for illustration)
    for iteration in range(10):
        print(f"\nIteration {iteration}")
        
        # Self-play
        games = trainer._run_self_play_batch(config.training.games_per_iteration)
        avg_reward = sum(g.rewards[0] for g in games) / len(games)
        print(f"  Avg reward: {avg_reward:.3f}")
        
        # Train
        losses = trainer._train_on_batch(games)
        print(f"  Policy loss: {losses['policy_loss']:.4f}")
        print(f"  Value loss: {losses['value_loss']:.4f}")
        
        # Periodic evaluation
        if iteration % 5 == 0:
            evaluator = PokerEvaluator(trainer.agent, config)
            results = evaluator.evaluate_vs_random(num_games=20)
            print(f"  Eval win rate: {results['win_rate']:.3f}")


# ============================================================================
# Run examples
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run example workflows")
    parser.add_argument("--example", type=int, default=1,
                       help="Example to run (1-6)")
    args = parser.parse_args()
    
    examples = {
        1: ("Basic Training", example_basic_training),
        2: ("Larger Model", example_larger_model),
        3: ("Ablation: No Transition", example_ablation_no_transition),
        4: ("Belief Analysis", example_belief_analysis),
        5: ("Search Comparison", example_search_comparison),
        6: ("Custom Training", example_custom_training),
    }
    
    if args.example in examples:
        name, func = examples[args.example]
        print(f"\n{'='*50}")
        print(f"Example {args.example}: {name}")
        print(f"{'='*50}\n")
        func()
    else:
        print("Available examples:")
        for idx, (name, _) in examples.items():
            print(f"  {idx}: {name}")
        print("\nRun with: python examples.py --example <number>")
