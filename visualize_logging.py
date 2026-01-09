#!/usr/bin/env python3
"""
Enhanced Pygame visualization with detailed move logging and evaluation.

This version adds comprehensive logging of every move with:
- Model policy probabilities and reasoning
- Belief state analysis
- Value function estimates
- Game outcome evaluation
- Performance statistics
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pygame
from src.config.config import ExperimentConfig, ModelConfig
from src.model import PokerTransformerAgent
from src.environment import KuhnPoker, Action, ObservableState
from src.visualization import PygamePokerVisualizer


class EnhancedPokerVisualizer(PygamePokerVisualizer):
    """Enhanced visualizer with detailed logging and evaluation."""
    
    def __init__(self, agent, config, **kwargs):
        super().__init__(agent, config, **kwargs)
        
        # Enhanced logging state
        self.move_count = 0
        self.game_move_log = []
        self.game_statistics = {
            'total_games': 0,
            'p0_wins': 0,
            'p1_wins': 0,
            'draws': 0,
            'total_moves': 0,
            'fold_count': 0,
            'call_count': 0,
            'raise_count': 0,
            'check_count': 0,
            'avg_value_p0': [],
            'avg_value_p1': [],
            'decision_quality': []
        }
        
        print("📊 Enhanced Logging Initialized")
        print("=" * 60)
    
    def log_move(self, player_id, action, belief, value, policy_probs, legal_actions, game_state):
        """Log detailed information about each move."""
        self.move_count += 1
        self.game_statistics['total_moves'] += 1
        
        # Count action types
        if action == Action.FOLD:
            self.game_statistics['fold_count'] += 1
        elif action == Action.CALL:
            self.game_statistics['call_count'] += 1
        elif action == Action.RAISE:
            self.game_statistics['raise_count'] += 1
        elif action == Action.CHECK:
            self.game_statistics['check_count'] += 1
        
        # Convert tensors to numpy for logging
        if hasattr(belief, 'cpu'):
            belief_np = belief.cpu().numpy()
        else:
            belief_np = np.array(belief)
            
        if hasattr(value, 'cpu'):
            value_val = value.cpu().item()
        else:
            value_val = float(value)
            
        if hasattr(policy_probs, 'cpu'):
            policy_np = policy_probs.cpu().numpy()
        else:
            policy_np = np.array(policy_probs)
        
        # Card information
        card_names = ['J', 'Q', 'K']
        player_card = card_names[game_state.private_cards[player_id]]
        
        # Create move log entry
        move_entry = {
            'move_number': self.move_count,
            'player': player_id,
            'player_card': player_card,
            'action': action.name,
            'action_value': action.value,
            'value_estimate': value_val,
            'belief_mean': float(np.mean(belief_np)),
            'belief_std': float(np.std(belief_np)),
            'policy_max': float(np.max(policy_np)),
            'policy_entropy': float(-np.sum(policy_np * np.log(policy_np + 1e-8))),
            'legal_actions': [a.name for a in legal_actions],
            'pot': game_state.pot,
            'stack': game_state.stacks[player_id]
        }
        
        self.game_move_log.append(move_entry)
        
        # Print detailed move log
        print(f"\n🎰 MOVE #{self.move_count} - Player {player_id}")
        print(f"   Card: {player_card}")
        print(f"   Action: {action.name}")
        print(f"   Value Estimate: {value_val:+.3f}")
        print(f"   Policy: ", end="")
        
        # Show policy probabilities
        action_names = ['FOLD', 'CALL', 'RAISE', 'CHECK']
        for i, prob in enumerate(policy_np):
            if Action(i) in legal_actions:
                print(f"{action_names[i]}({prob:.2f}) ", end="")
        print()
        
        # Belief state analysis
        print(f"   Belief State: mean={move_entry['belief_mean']:.3f}, std={move_entry['belief_std']:.3f}")
        print(f"   Policy Entropy: {move_entry['policy_entropy']:.3f} (higher = more uncertain)")
        print(f"   Pot: {game_state.pot} chips, Stack: {game_state.stacks[player_id]}")
        
        # Decision quality assessment
        legal_mask = np.zeros(4)
        for legal_action in legal_actions:
            legal_mask[legal_action.value] = 1
        
        masked_policy = policy_np * legal_mask
        if masked_policy.sum() > 0:
            masked_policy = masked_policy / masked_policy.sum()
            chosen_prob = masked_policy[action.value]
            
            # Assess decision quality
            if chosen_prob >= 0.4:
                quality = "CONFIDENT"
            elif chosen_prob >= 0.2:
                quality = "MODERATE"
            else:
                quality = "UNCERTAIN"
            
            print(f"   Decision Quality: {quality} (chosen action prob: {chosen_prob:.2f})")
            self.game_statistics['decision_quality'].append(chosen_prob)
    
    def step_game(self, action: Action):
        """Override to add logging to game stepping."""
        if self.game_over:
            return
        
        player_id = self.game_state.current_player
        
        # Get model predictions and select action before stepping
        with torch.no_grad():
            # Get current belief and predictions
            if hasattr(self, 'current_belief') and self.current_belief is not None:
                belief = self.current_belief
                value = self.agent.predict_value(belief)[0, 0]
                policy_logits = self.agent.predict_policy(belief)[0]
                policy_probs = torch.softmax(policy_logits, dim=-1)
            else:
                # Initialize belief for first move
                belief, _ = self.agent.encode_belief([self.obs])
                value = self.agent.predict_value(belief)[0, 0]
                policy_logits = self.agent.predict_policy(belief)[0]
                policy_probs = torch.softmax(policy_logits, dim=-1)
            
            # Select action using model policy (same as base visualizer)
            legal_actions = self.env.get_legal_actions(player_id)
            if len(legal_actions) == 0:
                action = Action.CHECK  # fallback
            else:
                # Create mask for legal actions
                legal_mask = torch.zeros(4, device=policy_logits.device)
                for a in legal_actions:
                    legal_mask[a.value] = 1.0
                # Mask illegal actions with large negative
                masked_logits = policy_logits + (1 - legal_mask) * (-1e9)
                action_idx = torch.argmax(masked_logits).item()
                action = Action(action_idx)
            
            # Log the move with the selected action
            self.log_move(player_id, action, belief, value, policy_probs, legal_actions, self.game_state)
        
        # Continue with original step logic
        super().step_game(action)
    
    def reset_game(self):
        """Override to log game completion."""
        # Log game results if we have a completed game
        if hasattr(self, 'game_state') and self.game_state.is_terminal:
            self.log_game_results()
        
        # Reset game state
        super().reset_game()
        self.move_count = 0
        self.game_move_log = []
    
    def log_game_results(self):
        """Log comprehensive results of completed game."""
        self.game_statistics['total_games'] += 1
        
        card_names = ['J', 'Q', 'K']
        p0_card = card_names[self.game_state.private_cards[0]]
        p1_card = card_names[self.game_state.private_cards[1]]
        
        print(f"\n🏁 GAME #{self.game_statistics['total_games']} RESULTS")
        print("=" * 50)
        print(f"Final Cards: P0={p0_card}, P1={p1_card}")
        
        if self.game_state.payoffs:
            p0_result = self.game_state.payoffs[0]
            p1_result = self.game_state.payoffs[1]
            
            print(f"Payoffs: P0 {p0_result:+.1f}, P1 {p1_result:+.1f}")
            
            # Update win statistics
            if p0_result > 0:
                winner = "Player 0"
                self.game_statistics['p0_wins'] += 1
            elif p1_result > 0:
                winner = "Player 1"
                self.game_statistics['p1_wins'] += 1
            else:
                winner = "Draw"
                self.game_statistics['draws'] += 1
            
            print(f"Winner: {winner}")
        else:
            print("Game ended without payoffs")
        
        # Log move summary
        print(f"Total Moves: {len(self.game_move_log)}")
        if self.game_move_log:
            avg_value = np.mean([m['value_estimate'] for m in self.game_move_log])
            avg_entropy = np.mean([m['policy_entropy'] for m in self.game_move_log])
            avg_quality = np.mean([m['policy_max'] for m in self.game_move_log])
            
            print(f"Average Value: {avg_value:+.3f}")
            print(f"Average Policy Entropy: {avg_entropy:.3f}")
            print(f"Average Decision Confidence: {avg_quality:.3f}")
        
        print("=" * 50)
    
    def print_overall_statistics(self):
        """Print comprehensive statistics across all games."""
        print(f"\n📊 OVERALL STATISTICS ({self.game_statistics['total_games']} games)")
        print("=" * 60)
        
        if self.game_statistics['total_games'] > 0:
            p0_winrate = self.game_statistics['p0_wins'] / self.game_statistics['total_games'] * 100
            p1_winrate = self.game_statistics['p1_wins'] / self.game_statistics['total_games'] * 100
            drawrate = self.game_statistics['draws'] / self.game_statistics['total_games'] * 100
            
            print(f"Win Rates:")
            print(f"  Player 0: {p0_winrate:.1f}% ({self.game_statistics['p0_wins']}/{self.game_statistics['total_games']})")
            print(f"  Player 1: {p1_winrate:.1f}% ({self.game_statistics['p1_wins']}/{self.game_statistics['total_games']})")
            print(f"  Draws: {drawrate:.1f}% ({self.game_statistics['draws']}/{self.game_statistics['total_games']})")
            
            print(f"\nAction Distribution:")
            total_actions = (self.game_statistics['fold_count'] + 
                          self.game_statistics['call_count'] + 
                          self.game_statistics['raise_count'] + 
                          self.game_statistics['check_count'])
            
            if total_actions > 0:
                print(f"  FOLD: {self.game_statistics['fold_count']/total_actions*100:.1f}%")
                print(f"  CALL: {self.game_statistics['call_count']/total_actions*100:.1f}%")
                print(f"  RAISE: {self.game_statistics['raise_count']/total_actions*100:.1f}%")
                print(f"  CHECK: {self.game_statistics['check_count']/total_actions*100:.1f}%")
            
            print(f"\nDecision Quality:")
            if self.game_statistics['decision_quality']:
                avg_confidence = np.mean(self.game_statistics['decision_quality'])
                print(f"  Average Confidence: {avg_confidence:.3f}")
                
                # Decision quality assessment
                if avg_confidence >= 0.6:
                    quality = "HIGHLY CONFIDENT"
                elif avg_confidence >= 0.4:
                    quality = "MODERATELY CONFIDENT"
                else:
                    quality = "UNCERTAIN/HESITANT"
                print(f"  Overall Quality: {quality}")
            
            print(f"Average Moves per Game: {self.game_statistics['total_moves']/self.game_statistics['total_games']:.1f}")
        
        print("=" * 60)
    
    def run(self):
        """Override to add final statistics output."""
        try:
            super().run()
        finally:
            # Always print final statistics when done
            self.print_overall_statistics()


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
        description="Enhanced Pygame visualization with detailed logging"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='poker_bot_final_model.pt',
        help='Path to trained model checkpoint'
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
        help='Delay between actions (ms, default: 2000)'
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
    print("🎮 Enhanced Poker Transformer Visualization")
    print("📊 With Detailed Move Logging & Evaluation")
    print("=" * 60)
    
    # Load model
    agent, config = load_model_with_config(args.model)
    
    if agent is None:
        print("Failed to load model. Exiting...")
        return
    
    # Create enhanced visualizer
    try:
        visualizer = EnhancedPokerVisualizer(
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
        
        print("Controls:")
        print("  R - Reset game")
        print("  A - Toggle auto-play mode")
        print("  Space - Single step (auto mode)")
        print("  Mouse - Click action buttons (manual mode)")
        print("  Q - Quit")
        print()
        print("📊 Enhanced Features:")
        print("  • Every move logged with detailed analysis")
        print("  • Policy probabilities and value estimates")
        print("  • Belief state dynamics")
        print("  • Decision quality assessment")
        print("  • Comprehensive game statistics")
        print()
        
        visualizer.run()
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()