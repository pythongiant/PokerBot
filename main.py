#!/usr/bin/env python3
"""
Main entry point for Poker Transformer research agent.

Example usage:
    python main.py --num-iterations 100 --batch-size 32 --latent-dim 64
    python main.py --config configs/default.yaml
"""

import argparse
import torch
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.config import ExperimentConfig, ModelConfig, TrainingConfig
from src.training import PokerTrainer
from src.evaluation import (
    PokerEvaluator,
    play_and_visualize_sample_game,
    visualize_geometry
)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train Poker Transformer with belief state + value + policy"
    )
    
    # Experiment
    parser.add_argument('--name', type=str, default='poker_transformer_default',
                       help='Experiment name')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Logging directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu or cuda')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Model architecture
    parser.add_argument('--latent-dim', type=int, default=64,
                       help='Latent belief state dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Transformer attention heads')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='Transformer layers')
    parser.add_argument('--ff-dim', type=int, default=256,
                       help='Feed-forward hidden dimension')
    parser.add_argument('--transition-type', type=str, default='deterministic',
                       help='Transition model: deterministic or probabilistic')
    
    # Training
    parser.add_argument('--num-iterations', type=int, default=100,
                       help='Number of training iterations')
    parser.add_argument('--games-per-iteration', type=int, default=128,
                       help='Games per training iteration')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       help='Optimizer: adam or sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Search
    parser.add_argument('--search-type', type=str, default='mcts',
                       help='Search type: mcts or rollout')
    parser.add_argument('--num-simulations', type=int, default=50,
                       help='MCTS simulations per position')
    parser.add_argument('--rollout-depth', type=int, default=10,
                       help='Rollout depth')
    
    # Evaluation
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation after training')
    parser.add_argument('--eval-games', type=int, default=100,
                       help='Number of evaluation games')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Load checkpoint and eval')
    
    return parser


def main():
    """Main entry point."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        name=args.name,
        log_dir=args.log_dir,
        device=args.device,
        seed=args.seed,
        model=ModelConfig(
            latent_dim=args.latent_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            transition_type=args.transition_type,
        ),
        training=TrainingConfig(
            num_iterations=args.num_iterations,
            games_per_iteration=args.games_per_iteration,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            search_type=args.search_type,
            num_simulations=args.num_simulations,
            rollout_depth=args.rollout_depth,
        ),
    )
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Poker Transformer Training")
    logger.info(f"Config: {config.to_dict()}")
    
    # Create trainer and run
    trainer = PokerTrainer(config)
    trainer.train()
    
    log_dir = Path(config.log_dir) / config.name
    
    # Optional: Evaluation
    if args.eval:
        logger.info("Running evaluation...")
        evaluator = PokerEvaluator(trainer.agent, config)
        eval_results = evaluator.run_full_evaluation()
        logger.info(f"Evaluation results: {eval_results}")
    
    # Play sample games and visualize
    logger.info("Playing sample games...")
    play_and_visualize_sample_game(trainer.agent, config, log_dir, num_games=2)
    
    # Visualize belief state geometry
    logger.info("Analyzing belief state geometry...")
    visualize_geometry(trainer.agent, config, log_dir)
    
    # Optional: Load checkpoint and evaluate
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        trainer.agent.load_state_dict(checkpoint['agent_state'])
        
        evaluator = PokerEvaluator(trainer.agent, config)
        eval_results = evaluator.run_full_evaluation()
        logger.info(f"Evaluation results: {eval_results}")
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
