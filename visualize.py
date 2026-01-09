#!/usr/bin/env python3
"""
Interactive Pygame visualization of Poker Transformer Agent.

Shows real-time activation visualizations of:
- Belief states (64-dimensional latent vector)
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
from src.config import DEFAULT_CONFIG
from src.model import PokerTransformerAgent
from src.visualization import PygamePokerVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Pygame visualization of Poker Transformer Agent"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='logs/poker_transformer_default/checkpoint_iter40.pt',
        help='Path to trained model checkpoint (uses latest pretrained if not provided)'
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
    if args.human:
        print("Poker: Human vs AI Model")
        print(f"You are Player {args.human_player} (0 or 1)")
    else:
        print("Poker Transformer Agent - Pygame Visualization")
    print("=" * 60)
    
    config = DEFAULT_CONFIG
    print(f"Model config: latent_dim={config.model.latent_dim}, "
          f"num_layers={config.model.num_layers}")
    
    agent = PokerTransformerAgent(config)
    
    if args.model:
        if os.path.exists(args.model):
            try:
                checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
                agent.load_state_dict(checkpoint['agent_state'])
                print(f"✓ Loaded model from {args.model}")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                print("Using randomly initialized model instead")
        else:
            print(f"✗ Model file not found: {args.model}")
            print("Using randomly initialized model instead")
    else:
        print("Using randomly initialized model")
    
    visualizer = PygamePokerVisualizer(
        agent,
        config,
        width=args.width,
        height=args.height,
        human_vs_model=args.human,
        human_player=args.human_player,
        auto_continue=not args.no_auto_continue,
        alternate=args.alternate,
        random_ai=args.random_ai
    )
    
    if args.auto:
        visualizer.auto_play = True
        print("Auto-play mode enabled")
    
    visualizer.play_delay = args.play_delay
    
    print("\nControls:")
    print("  R - Reset game")
    if args.human:
        print("  Mouse - Click action buttons to make your moves")
        if args.alternate:
            print("  Alternating: Who starts changes each game")
        else:
            print(f"  Fixed: You play as P{args.human_player} every game")
        if args.random_ai:
            print("  Random AI: AI plays randomly (for testing)")
        else:
            print("  Trained AI: AI uses learned poker strategy")
    else:
        print("  A - Toggle auto-play mode")
        print("  Space - Single step (auto mode)")
        print("  Mouse - Click action buttons (manual mode)")
    print("  Q - Quit")
    print("\nStarting visualization...")
    print("-" * 60)
    
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Visualization closed")


if __name__ == '__main__':
    main()
