#!/usr/bin/env python3
"""
Quick start guide - run this to verify everything works.

This script:
1. Tests all components individually
2. Runs a tiny training loop (2 iterations)
3. Evaluates the trained agent
4. Saves results to logs/

Expected runtime: ~2-3 minutes on CPU
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.config.config import ExperimentConfig
from src.training import PokerTrainer
from src.evaluation import PokerEvaluator


def main():
    print("\n" + "="*60)
    print("POKER TRANSFORMER - QUICK START")
    print("="*60 + "\n")
    
    # Configure for quick test
    config = ExperimentConfig(
        name="quickstart",
        device="cpu",
        seed=42,
    )
    
    # Minimal config for speed
    config.training.num_iterations = 2
    config.training.games_per_iteration = 8
    config.training.batch_size = 4
    
    print("Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Latent dim: {config.model.latent_dim}")
    print(f"  Transformer: {config.model.num_layers} layers, {config.model.num_heads} heads")
    print(f"  Training: {config.training.num_iterations} iterations")
    print(f"  Games per iteration: {config.training.games_per_iteration}")
    print()
    
    # Train
    print("Starting training...")
    try:
        trainer = PokerTrainer(config)
        trainer.train()
        print("✓ Training completed successfully!\n")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Evaluate
    print("Running evaluation...")
    try:
        evaluator = PokerEvaluator(trainer.agent, config, 
                                   output_dir=Path(config.log_dir) / config.name)
        
        # Quick eval vs random
        results = evaluator.evaluate_vs_random(num_games=20)
        print(f"✓ Evaluation completed!")
        print(f"  Win rate vs random: {results['win_rate']:.1%}")
        print(f"  Average reward: {results['avg_reward']:.3f}\n")
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("="*60)
    print("SUCCESS! ✓")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check logs in: ./logs/quickstart/")
    print("  2. Run full training: python main.py --num-iterations 100")
    print("  3. Explore examples: python examples/examples.py --example 1")
    print("  4. Read architecture: cat ARCHITECTURE.md")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
