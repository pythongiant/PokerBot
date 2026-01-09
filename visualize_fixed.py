#!/usr/bin/env python3
"""
Interactive Pygame visualization of Poker Transformer Agent with proper model loading.

Shows real-time activation visualizations of:
- Belief states (latent vector from checkpoint)
- Attention weights (how model attends to history)
- Value function (expected payoff)
- Transition model (predicted vs actual belief updates)

Controls:
- R: Reset game
- A: Toggle auto-play mode
- Space: Single step (when in auto mode)
- Q: Quit
- Mouse: Click action buttons in manual mode
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config.config import ExperimentConfig, ModelConfig
from src.model import PokerTransformerAgent
from src.visualization import PygamePokerVisualizer


def load_model_with_config(checkpoint_path):
    """Load model with architecture from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract model config from checkpoint if available
        if 'config' in checkpoint and hasattr(checkpoint['config'], 'model'):
            model_config_obj = checkpoint['config'].model
            model_config = ModelConfig(
                latent_dim=getattr(model_config_obj, 'latent_dim', 128),
                num_heads=getattr(model_config_obj, 'num_heads', 8),
                num_layers=getattr(model_config_obj, 'num_layers', 4),
                ff_dim=getattr(model_config_obj, 'ff_dim', 256),
                transition_type=getattr(model_config_obj, 'transition_type', 'deterministic'),
                value_hidden_dim=getattr(model_config_obj, 'value_hidden_dim', 128),
                policy_hidden_dim=getattr(model_config_obj, 'policy_hidden_dim', 128)
            )
            print(f"Model config from checkpoint: latent_dim={model_config.latent_dim}, num_layers={model_config.num_layers}, num_heads={model_config.num_heads}")
        else:
            # Use our trained model's known config
            model_config = ModelConfig(
                latent_dim=128,
                num_heads=8,
                num_layers=4,
                ff_dim=256,
                transition_type='deterministic',
                value_hidden_dim=128,
                policy_hidden_dim=128
            )
            print("Using known model config (128dim, 8heads, 4layers)")
        
        # Create full config
        config = ExperimentConfig(
            name='poker_final_model',
            model=model_config
        )
        
        # Create agent and load weights
        agent = PokerTransformerAgent(config)
        agent.load_state_dict(checkpoint['agent_state'], strict=False)
        agent.eval()
        
        print("✅ Model loaded successfully!")
        return agent, config
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Pygame visualization of Poker Transformer Agent"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='poker_bot_final_model.pt',
        help='Path to trained model checkpoint (default: poker_bot_final_model.pt)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1400,
        help='Window width (default: 1400)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=900,
        help='Window height (default: 900)'
    )
    parser.add_argument(
        '--auto', 
        action='store_true', 
        help='Start in auto-play mode'
    )
    parser.add_argument(
        '--play-delay',
        type=int,
        default=2000,
        help='Delay between actions (ms, default: 2000 - use 3000 for slower back-and-forth)'
    )
    parser.add_argument(
        '--human',
        action='store_true',
        help='Play as human vs AI model'
    )
    parser.add_argument(
        '--human-player',
        type=int,
        choices=[0, 1],
        default=0,
        help='Which player is human (0 or 1, default: 0)'
    )
    parser.add_argument(
        '--alternate',
        action='store_true',
        help='Alternate who starts each game (model/human)'
    )
    parser.add_argument(
        '--no-auto-continue',
        action='store_true',
        help='Disable automatic game continuation in human vs AI mode'
    )
    parser.add_argument(
        '--random-ai',
        action='store_true',
        help='Use random policy for AI instead of trained model'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Poker Transformer Agent - Pygame Visualization")
    print("=" * 60)
    
    # Load model
    agent, config = load_model_with_config(args.model)
    
    if agent is None:
        print("Failed to load model. Exiting...")
        return
    
    if config and hasattr(config, 'model'):
        print(f"Model config: latent_dim={config.model.latent_dim}, num_layers={config.model.num_layers}, num_heads={config.model.num_heads}")
    
    # Create visualizer
    try:
        visualizer = PygamePokerVisualizer(
            agent=agent,
            config=config,
            width=args.width,
            height=args.height,
            human_vs_model=args.human,
            human_player=args.human_player if args.human else 0,
            auto_continue=not args.no_auto_continue,
            alternate=args.alternate,
            random_ai=args.random_ai
        )
        
        # Set auto mode if requested
        if args.auto:
            visualizer.auto_play = True
        
        print("Starting visualization...")
        print("-" * 60)
        print("Controls:")
        print("  R - Reset game")
        print("  A - Toggle auto-play mode")
        print("  Space - Single step (auto mode)")
        print("  Mouse - Click action buttons (manual mode)")
        print("  Q - Quit")
        print("-" * 60)
        print("📊 ENHANCED LOGGING ENABLED:")
        print("  • Every move will be logged with model reasoning")
        print("  • Policy probabilities and value estimates")
        print("  • Belief state analysis")
        print("  • Game outcome evaluation")
        print("-" * 60)
        
        # Add logging functionality to visualizer
        visualizer.enable_detailed_logging = True
        visualizer.run()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("Visualization closed")


if __name__ == '__main__':
    main()