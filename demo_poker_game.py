#!/usr/bin/env python3
"""
Demo version of Text-based Poker Game that plays automatically
to showcase the model's behavior and analysis features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
import torch
from src.config.config import ExperimentConfig, ModelConfig
from src.model import PokerTransformerAgent
from src.environment.kuhn import KuhnPoker, Action


class DemoPokerGame:
    """Demo version that plays automatically to showcase model behavior."""
    
    def __init__(self, model_path='poker_bot_final_model.pt'):
        self.model_path = model_path
        self.env = KuhnPoker(seed=42)
        self.game_stats = {
            'games_played': 0,
            'player0_wins': 0,
            'player1_wins': 0,
            'draws': 0,
            'total_moves': 0,
            'action_counts': {'FOLD': 0, 'CALL': 0, 'RAISE': 0, 'CHECK': 0},
            'call_opportunities': 0,
            'calls_made': 0
        }
        
        # Load model
        self.load_model()
        
        # Card names for display
        self.card_names = ['J', 'Q', 'K']
        
        print("🎮 DEMO: TEXT-BASED POKER GAME")
        print("=" * 60)
        print("Showcasing trained AI model behavior in Kuhn Poker")
        print(f"Model: {self.model_path}")
        print("=" * 60)
    
    def load_model(self):
        """Load the trained poker model."""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            config = ExperimentConfig(
                name='poker_final_model',
                model=ModelConfig(
                    latent_dim=128,
                    num_heads=8,
                    num_layers=4,
                    ff_dim=256,
                    transition_type='deterministic'
                )
            )
            self.agent = PokerTransformerAgent(config)
            self.agent.load_state_dict(checkpoint['agent_state'], strict=False)
            self.agent.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def get_model_action(self, game_state, obs, player_id):
        """Get model's action with detailed analysis."""
        with torch.no_grad():
            belief, _ = self.agent.encode_belief([obs])
            value = self.agent.predict_value(belief)[0, 0].item()
            policy_logits = self.agent.predict_policy(belief)[0]
            policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
            
            legal_actions = self.env.get_legal_actions(player_id)
            
            # Select action using model policy
            if len(legal_actions) == 0:
                action = Action.CHECK
            else:
                legal_mask = torch.zeros(4, device=policy_logits.device)
                for a in legal_actions:
                    legal_mask[a.value] = 1.0
                masked_logits = policy_logits + (1 - legal_mask) * (-1e9)
                action_idx = torch.argmax(masked_logits).item()
                action = Action(action_idx)
            
            return action, policy_probs, value, belief
    
    def display_game_state(self, game_state, p0_card, p1_card, current_player):
        """Display current game state."""
        print(f"\n{'='*60}")
        print(f"🎰 GAME #{self.game_stats['games_played'] + 1} - Move #{self.game_stats['total_moves'] + 1}")
        print(f"{'='*60}")
        
        print(f"\n📋 GAME STATUS:")
        print(f"  Pot: {game_state.pot} chips")
        print(f"  Player 0 stack: {game_state.stacks[0]} chips")
        print(f"  Player 1 stack: {game_state.stacks[1]} chips")
        
        print(f"\n🃏 CARDS:")
        print(f"  Player 0: {p0_card}")
        print(f"  Player 1: {p1_card}")
        
        print(f"\n🎯 CURRENT TURN:")
        print(f"  Player {current_player} to act")
        
        # Show betting history
        if any(game_state.public_bets[0]) or any(game_state.public_bets[1]):
            print(f"\n📜 BETTING HISTORY:")
            for player_idx, bets in enumerate(game_state.public_bets):
                if bets:
                    for action, amount in bets:
                        print(f"  Player {player_idx}: {action.name}")
    
    def analyze_model_move(self, player_id, action, policy_probs, value, card, pot, legal_actions):
        """Display detailed analysis of model's move."""
        print(f"\n🤖 PLAYER {player_id} ANALYSIS:")
        print(f"  Card: {card}")
        print(f"  Pot: {pot} chips")
        print(f"  Value estimate: {value:+.3f}")
        
        print(f"\n📊 MODEL POLICY:")
        action_names = ['FOLD', 'CALL', 'RAISE', 'CHECK']
        for i, prob in enumerate(policy_probs):
            if Action(i) in legal_actions:
                status = "✅ CHOSEN" if Action(i) == action else "  "
                print(f"  {status} {action_names[i]:8s}: {prob:.4f}")
        
        # Check if this was a CALL opportunity
        if Action.CALL in legal_actions:
            self.game_stats['call_opportunities'] += 1
            if action == Action.CALL:
                self.game_stats['calls_made'] += 1
                print(f"\n🎯 CALL ANALYSIS: ✅ Model chose to CALL!")
            else:
                print(f"\n🎯 CALL ANALYSIS: ❌ Model did NOT call (chose {action.name})")
        
        # Assess move quality
        print(f"\n🎯 MOVE ASSESSMENT:")
        print(f"  Chose: {action.name}")
        
        # Strategic assessment
        if action == Action.FOLD:
            print("  Strategy: Conservative - giving up to save chips")
        elif action == Action.CALL:
            print("  Strategy: Moderate - matching opponent's bet")
        elif action == Action.RAISE:
            print("  Strategy: Aggressive - increasing the pot")
        elif action == Action.CHECK:
            print("  Strategy: Passive - passing action to opponent")
        
        # Confidence assessment
        chosen_prob = policy_probs[action.value]
        if chosen_prob >= 0.4:
            confidence = "HIGHLY CONFIDENT"
        elif chosen_prob >= 0.25:
            confidence = "MODERATELY CONFIDENT"
        else:
            confidence = "UNCERTAIN"
        
        print(f"  Confidence: {confidence} ({chosen_prob:.2f} probability)")
        
        # Simple strategic advice
        if Action.CALL in legal_actions and action != Action.CALL:
            print(f"  💡 STRATEGIC NOTE: CALL was available with {policy_probs[Action.CALL.value]:.2f} probability")
    
    def display_game_result(self, game_state, p0_card, p1_card):
        """Display final game result."""
        print(f"\n🏁 GAME OVER!")
        print(f"{'='*60}")
        
        print(f"\n🃏 FINAL CARDS:")
        print(f"  Player 0: {p0_card}")
        print(f"  Player 1: {p1_card}")
        
        if game_state.payoffs:
            p0_result = game_state.payoffs[0]
            p1_result = game_state.payoffs[1]
            
            print(f"\n💰 RESULTS:")
            print(f"  Player 0: {p0_result:+.1f} chips")
            print(f"  Player 1: {p1_result:+.1f} chips")
            
            if p0_result > 0:
                winner = "PLAYER 0 WINS!"
                self.game_stats['player0_wins'] += 1
            elif p1_result > 0:
                winner = "PLAYER 1 WINS!"
                self.game_stats['player1_wins'] += 1
            else:
                winner = "DRAW!"
                self.game_stats['draws'] += 1
            
            print(f"\n🏆 {winner}")
        else:
            print("\n❓ No clear result determined")
        
        # Update statistics
        self.game_stats['games_played'] += 1
    
    def play_demo_game(self):
        """Play a single demo game with both players using the model."""
        # Reset game
        game_state, obs = self.env.reset()
        
        # Get card assignments
        p0_card_idx = game_state.private_cards[0]
        p1_card_idx = game_state.private_cards[1]
        p0_card = self.card_names[p0_card_idx]
        p1_card = self.card_names[p1_card_idx]
        
        move_count = 0
        max_moves = 10
        
        print(f"\n🎮 Starting demo game...")
        print(f"Both players will use the trained model")
        
        while not game_state.is_terminal and move_count < max_moves:
            move_count += 1
            self.game_stats['total_moves'] += 1
            
            current_player = game_state.current_player
            
            # Display game state
            self.display_game_state(game_state, p0_card, p1_card, current_player)
            
            # Get model action
            action, policy_probs, value, belief = self.get_model_action(game_state, obs, current_player)
            
            # Get player card for analysis
            player_card = p0_card if current_player == 0 else p1_card
            
            # Display analysis
            self.analyze_model_move(current_player, action, policy_probs, value, player_card, game_state.pot, self.env.get_legal_actions(current_player))
            
            # Record action
            self.game_stats['action_counts'][action.name] += 1
            
            print(f"\n🤖 Player {current_player} chose: {action.name}")
            
            # Step the game
            game_state, obs, _ = self.env.step(current_player, action, amount=1)
            
            # Brief pause for readability
            input("\nPress Enter to continue to next move...")
        
        # Display result
        self.display_game_result(game_state, p0_card, p1_card)
    
    def display_overall_stats(self):
        """Display overall game statistics."""
        print(f"\n📊 OVERALL STATISTICS")
        print(f"{'='*60}")
        
        if self.game_stats['games_played'] > 0:
            print(f"\n🎮 GAMES PLAYED: {self.game_stats['games_played']}")
            print(f"  Player 0 wins: {self.game_stats['player0_wins']} ({self.game_stats['player0_wins']/self.game_stats['games_played']*100:.1f}%)")
            print(f"  Player 1 wins: {self.game_stats['player1_wins']} ({self.game_stats['player1_wins']/self.game_stats['games_played']*100:.1f}%)")
            print(f"  Draws: {self.game_stats['draws']} ({self.game_stats['draws']/self.game_stats['games_played']*100:.1f}%)")
            
            print(f"\n🎯 ACTION ANALYSIS:")
            total = sum(self.game_stats['action_counts'].values())
            for action, count in self.game_stats['action_counts'].items():
                percentage = count / total * 100
                print(f"  {action}: {count} ({percentage:.1f}%)")
            
            print(f"\n📞 CALLING BEHAVIOR:")
            if self.game_stats['call_opportunities'] > 0:
                call_rate = self.game_stats['calls_made'] / self.game_stats['call_opportunities'] * 100
                print(f"  CALL opportunities: {self.game_stats['call_opportunities']}")
                print(f"  CALLs actually made: {self.game_stats['calls_made']}")
                print(f"  CALL rate: {call_rate:.1f}%")
                
                if call_rate < 5:
                    print(f"  🚨 Model severely under-calls (<5% rate)!")
                elif call_rate < 15:
                    print(f"  ⚠️  Model under-calls (<15% rate)")
                else:
                    print(f"  ✅ Model calls appropriately")
            else:
                print("  No CALL opportunities encountered")
            
            print(f"\n📈 Average moves per game: {self.game_stats['total_moves']/self.game_stats['games_played']:.1f}")
        else:
            print("No games played yet.")
    
    def run_demo(self):
        """Run demo games."""
        print("\n🎮 Welcome to Demo Poker Game!")
        print("This will showcase the trained model's behavior")
        print("Both players will use the same trained model")
        
        # Play multiple demo games
        num_games = 3
        for i in range(num_games):
            if i > 0:
                input(f"\nPress Enter to start game {i+1}...")
            
            self.play_demo_game()
        
        # Display final statistics
        self.display_overall_stats()
        
        print(f"\n🎮 Demo complete!")
        print("You can now run the interactive version with:")
        print("python text_poker_game.py")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo poker game showcasing AI behavior")
    parser.add_argument('--model', type=str, default='poker_bot_final_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--games', type=int, default=3,
                       help='Number of demo games to play')
    
    args = parser.parse_args()
    
    # Create and run demo
    game = DemoPokerGame(model_path=args.model)
    
    # Play specified number of games
    for i in range(args.games):
        if i > 0:
            input(f"\nPress Enter to start game {i+1}...")
        
        game.play_demo_game()
    
    # Display final statistics
    game.display_overall_stats()
    
    print(f"\n🎮 Demo complete!")
    print("You can now run the interactive version with:")
    print("python text_poker_game.py")


if __name__ == '__main__':
    main()