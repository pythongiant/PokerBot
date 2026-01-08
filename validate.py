"""
Validation script: Verify all components work together.

This script tests:
1. Environment (Kuhn poker) works correctly
2. Model can be instantiated and forward pass succeeds
3. Training loop runs for a few iterations
4. Evaluation metrics compute without errors
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.config import ExperimentConfig
from src.environment import KuhnPoker, Action
from src.model import PokerTransformerAgent
from src.training import PokerTrainer, run_self_play_game
from src.evaluation import PokerEvaluator


def test_environment():
    """Test Kuhn poker environment."""
    print("\n=== Testing Environment ===")
    
    env = KuhnPoker(seed=42)
    game_state, obs = env.reset()
    
    print(f"Initial observation: own_card={obs.own_card}")
    print(f"Legal actions: {env.get_legal_actions(0)}")
    
    # Play a simple game
    steps = 0
    while not game_state.is_terminal and steps < 10:
        legal_actions = env.get_legal_actions(game_state.current_player)
        action = legal_actions[0]  # Take first legal action
        game_state, obs, _ = env.step(game_state.current_player, Action(action), amount=1)
        steps += 1
    
    print(f"Game completed in {steps} steps")
    print(f"Final payoffs: {game_state.payoffs}")
    print("✓ Environment test passed")


def test_model():
    """Test model instantiation and forward pass."""
    print("\n=== Testing Model ===")
    
    config = ExperimentConfig()
    agent = PokerTransformerAgent(config)
    
    print(f"Model architecture:")
    print(f"  Belief encoder: {config.model.latent_dim}D latent")
    print(f"  Transformer: {config.model.num_layers} layers, {config.model.num_heads} heads")
    print(f"  Heads: value, policy, transition, opponent_range")
    
    # Create dummy observation
    from src.environment import ObservableState
    obs = ObservableState(
        public_cards=[],
        own_card=0,
        action_history=[(0, Action.CHECK, 0)],
        current_player=1,
        stacks=[100, 100],
        pot=2,
        street=0,
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = agent([obs])
    
    print(f"Output shapes:")
    print(f"  Belief states: {outputs['belief_states'].shape}")
    print(f"  Values: {outputs['values'].shape}")
    print(f"  Policy logits: {outputs['policy_logits'].shape}")
    
    print("✓ Model test passed")


def test_training_step():
    """Test one training iteration."""
    print("\n=== Testing Training ===")
    
    config = ExperimentConfig(
        training=ExperimentConfig().training
    )
    config.training.num_iterations = 2  # Just 2 iterations
    config.training.games_per_iteration = 4  # Small batch
    config.training.batch_size = 2
    
    trainer = PokerTrainer(config)
    
    print(f"Training config:")
    print(f"  Games per iteration: {config.training.games_per_iteration}")
    print(f"  Batch size: {config.training.batch_size}")
    
    # Run one iteration
    print("\nRunning self-play...")
    games = trainer._run_self_play_batch(4)
    print(f"  Generated {len(games)} games")
    
    # Train
    print("Training on batch...")
    losses = trainer._train_on_batch(games)
    print(f"  Losses: {losses}")
    
    print("✓ Training test passed")


def test_evaluation():
    """Test evaluation suite."""
    print("\n=== Testing Evaluation ===")
    
    config = ExperimentConfig()
    config.device = 'cpu'
    
    agent = PokerTransformerAgent(config)
    evaluator = PokerEvaluator(agent, config)
    
    print("Running vs. random baseline...")
    results = evaluator.evaluate_vs_random(num_games=10)
    print(f"  Win rate: {results['win_rate']:.3f}")
    print(f"  Avg reward: {results['avg_reward']:.3f}")
    
    print("Analyzing belief states...")
    belief_analysis = evaluator.analyze_belief_states(num_games=3)
    print(f"  Num samples: {belief_analysis['num_samples']}")
    print(f"  Mean belief: {belief_analysis['belief_mean'][:5]}...")  # First 5 dims
    
    print("✓ Evaluation test passed")


def test_self_play_game():
    """Test one complete self-play game."""
    print("\n=== Testing Self-Play Game ===")
    
    config = ExperimentConfig()
    env = KuhnPoker(seed=42)
    agent = PokerTransformerAgent(config)
    
    game = run_self_play_game(agent, env, searcher=None)
    
    print(f"Game completed:")
    print(f"  Steps: {len(game.actions)}")
    print(f"  P0 reward: {game.rewards[0]}")
    print(f"  P1 reward: {game.rewards[1]}")
    
    print("✓ Self-play test passed")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Poker Transformer Validation Suite")
    print("=" * 50)
    
    try:
        test_environment()
        test_model()
        test_self_play_game()
        test_training_step()
        test_evaluation()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
        print("\nYou can now run training with:")
        print("  python main.py --num-iterations 100")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
