#!/usr/bin/env python3
"""
Text-based Poker Game: Human vs Trained AI Model

Play Kuhn Poker against the trained transformer model with:
- Real-time move analysis
- Model reasoning display
- Game statistics
- Interactive gameplay
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
from src.config.config import ExperimentConfig, ModelConfig
from src.model import PokerTransformerAgent
from src.environment.kuhn import KuhnPoker, Action


class TextPokerGame:
    """Interactive text-based poker game against trained model."""
    
    def __init__(self, model_path='poker_bot_final_model.pt'):
        self.model_path = model_path
        self.env = KuhnPoker(seed=42)
        self.game_stats = {
            'games_played': 0,
            'human_wins': 0,
            'ai_wins': 0,
            'draws': 0,
            'human_chips': 0,
            'ai_chips': 0,
            'total_moves': 0,
            'action_counts': {'FOLD': 0, 'CALL': 0, 'RAISE': 0, 'CHECK': 0}
        }
        
        # Load model
        self.load_model()
        
        # Card names for display
        self.card_names = ['J', 'Q', 'K']
        self.card_ranks = {'J': 1, 'Q': 2, 'K': 3}
        
        print("🎮 TEXT-BASED POKER GAME")
        print("=" * 50)
        print("Playing Kuhn Poker against trained AI model")
        print(f"Model: {self.model_path}")
        print("=" * 50)
    
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
    
    def get_ai_action(self, game_state, obs, player_id):
        """Get AI model's action with detailed analysis."""
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
    
    def display_game_state(self, game_state, human_card, ai_card, is_human_turn):
        """Display current game state."""
        print(f"\n{'='*60}")
        print(f"🎰 GAME #{self.game_stats['games_played'] + 1}")
        print(f"{'='*60}")
        
        print(f"\n📋 GAME STATUS:")
        print(f"  Pot: {game_state.pot} chips")
        print(f"  Your stack: {game_state.stacks[0]} chips")
        print(f"  AI stack: {game_state.stacks[1]} chips")
        
        print(f"\n🃏 CARDS:")
        print(f"  Your card: {human_card} {'(hidden)' if not is_human_turn else '(visible)'}")
        print(f"  AI card: {'?' if is_human_turn else ai_card} {'(hidden)' if is_human_turn else '(revealed)'}")
        
        print(f"\n🎯 CURRENT TURN:")
        if is_human_turn:
            print("  It's YOUR turn to act!")
        else:
            print("  AI is thinking...")
        
        # Show betting history
        if any(game_state.public_bets[0]) or any(game_state.public_bets[1]):
            print(f"\n📜 BETTING HISTORY:")
            for player_idx, bets in enumerate(game_state.public_bets):
                if bets:
                    player_name = "You" if player_idx == 0 else "AI"
                    for action, amount in bets:
                        print(f"  {player_name}: {action.name}")
    
    def get_human_action(self, legal_actions):
        """Get human player's action."""
        print(f"\n🎮 YOUR OPTIONS:")
        
        action_map = {}
        for i, action in enumerate(legal_actions):
            action_map[str(i+1)] = action
            print(f"  {i+1}. {action.name}")
        
        while True:
            try:
                choice = input(f"\nChoose your action (1-{len(legal_actions)}): ").strip()
                if choice in action_map:
                    return action_map[choice]
                else:
                    print("❌ Invalid choice. Please try again.")
            except KeyboardInterrupt:
                print("\n👋 Thanks for playing!")
                sys.exit(0)
    
    def analyze_ai_move(self, action, policy_probs, value, ai_card, pot, legal_actions):
        """Display detailed analysis of AI's move."""
        print(f"\n🤖 AI ANALYSIS:")
        print(f"  AI card: {ai_card}")
        print(f"  Pot: {pot} chips")
        print(f"  AI value estimate: {value:+.3f}")
        
        print(f"\n📊 AI POLICY:")
        action_names = ['FOLD', 'CALL', 'RAISE', 'CHECK']
        for i, prob in enumerate(policy_probs):
            if Action(i) in legal_actions:
                status = "✅ CHOSEN" if Action(i) == action else "  "
                print(f"  {status} {action_names[i]:8s}: {prob:.4f}")
        
        # Assess move quality
        print(f"\n🎯 MOVE ASSESSMENT:")
        print(f"  AI chose: {action.name}")
        
        # Simple strategic assessment
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
    
    def display_game_result(self, game_state, human_card, ai_card):
        """Display final game result."""
        print(f"\n🏁 GAME OVER!")
        print(f"{'='*60}")
        
        print(f"\n🃏 FINAL CARDS:")
        print(f"  Your card: {human_card}")
        print(f"  AI card: {ai_card}")
        
        if game_state.payoffs:
            human_result = game_state.payoffs[0]
            ai_result = game_state.payoffs[1]
            
            print(f"\n💰 RESULTS:")
            print(f"  You: {human_result:+.1f} chips")
            print(f"  AI: {ai_result:+.1f} chips")
            
            if human_result > 0:
                winner = "YOU WIN!"
                self.game_stats['human_wins'] += 1
                self.game_stats['human_chips'] += human_result
            elif ai_result > 0:
                winner = "AI WINS!"
                self.game_stats['ai_wins'] += 1
                self.game_stats['ai_chips'] += ai_result
            else:
                winner = "DRAW!"
                self.game_stats['draws'] += 1
            
            print(f"\n🏆 {winner}")
        else:
            print("\n❓ No clear result determined")
        
        # Update statistics
        self.game_stats['games_played'] += 1
    
    def display_overall_stats(self):
        """Display overall game statistics."""
        print(f"\n📊 OVERALL STATISTICS")
        print(f"{'='*60}")
        
        if self.game_stats['games_played'] > 0:
            print(f"\n🎮 GAMES PLAYED: {self.game_stats['games_played']}")
            print(f"  Your wins: {self.game_stats['human_wins']} ({self.game_stats['human_wins']/self.game_stats['games_played']*100:.1f}%)")
            print(f"  AI wins: {self.game_stats['ai_wins']} ({self.game_stats['ai_wins']/self.game_stats['games_played']*100:.1f}%)")
            print(f"  Draws: {self.game_stats['draws']} ({self.game_stats['draws']/self.game_stats['games_played']*100:.1f}%)")
            
            print(f"\n💰 CHIP PERFORMANCE:")
            print(f"  Your net: {self.game_stats['human_chips']:+.1f} chips")
            print(f"  AI net: {self.game_stats['ai_chips']:+.1f} chips")
            
            if self.game_stats['total_moves'] > 0:
                print(f"\n🎯 ACTION ANALYSIS:")
                total = sum(self.game_stats['action_counts'].values())
                for action, count in self.game_stats['action_counts'].items():
                    percentage = count / total * 100
                    print(f"  {action}: {count} ({percentage:.1f}%)")
                
                print(f"\n📈 Average moves per game: {self.game_stats['total_moves']/self.game_stats['games_played']:.1f}")
        else:
            print("No games played yet.")
    
    def play_game(self, human_first=True):
        """Play a single game of poker."""
        # Reset game
        game_state, obs = self.env.reset()
        
        # Get card assignments
        human_card_idx = game_state.private_cards[0]
        ai_card_idx = game_state.private_cards[1]
        human_card = self.card_names[human_card_idx]
        ai_card = self.card_names[ai_card_idx]
        
        # Determine who goes first
        if not human_first:
            # AI goes first
            if game_state.current_player == 1:
                # Swap players so AI goes first
                game_state.private_cards = [ai_card_idx, human_card_idx]
                human_card_idx, ai_card_idx = ai_card_idx, human_card_idx
                human_card, ai_card = ai_card, human_card
        
        move_count = 0
        max_moves = 10
        
        while not game_state.is_terminal and move_count < max_moves:
            move_count += 1
            self.game_stats['total_moves'] += 1
            
            current_player = game_state.current_player
            is_human_turn = (current_player == 0)
            
            # Display game state
            self.display_game_state(game_state, human_card, ai_card, is_human_turn)
            
            if is_human_turn:
                # Human's turn
                legal_actions = self.env.get_legal_actions(0)
                action = self.get_human_action(legal_actions)
                print(f"\n👤 You chose: {action.name}")
                
            else:
                # AI's turn
                legal_actions = self.env.get_legal_actions(1)
                action, policy_probs, value, belief = self.get_ai_action(game_state, obs, 1)
                
                # Display AI analysis
                self.analyze_ai_move(action, policy_probs, value, ai_card, game_state.pot, legal_actions)
                
                print(f"\n🤖 AI chose: {action.name}")
            
            # Record action
            self.game_stats['action_counts'][action.name] += 1
            
            # Step the game
            game_state, obs, _ = self.env.step(current_player, action, amount=1)
            
            # Brief pause for readability
            if not is_human_turn:
                input("\nPress Enter to continue...")
        
        # Display result
        self.display_game_result(game_state, human_card, ai_card)
    
    def run(self):
        """Main game loop."""
        print("\n🎮 Welcome to Text-Based Poker!")
        print("You'll be playing Kuhn Poker against a trained AI model.")
        print("\nGame Rules:")
        print("• Each player gets one card (J, Q, or K)")
        print("• K beats Q beats J")
        print("• Players can CHECK, CALL, RAISE, or FOLD")
        print("• Winner takes the pot!")
        
        while True:
            print(f"\n{'='*60}")
            print("🎮 MAIN MENU")
            print(f"{'='*60}")
            print("1. Play game (you go first)")
            print("2. Play game (AI goes first)")
            print("3. View statistics")
            print("4. Quit")
            
            choice = input("\nChoose option (1-4): ").strip()
            
            if choice == '1':
                self.play_game(human_first=True)
            elif choice == '2':
                self.play_game(human_first=False)
            elif choice == '3':
                self.display_overall_stats()
            elif choice == '4':
                print("\n👋 Thanks for playing!")
                self.display_overall_stats()
                break
            else:
                print("❌ Invalid choice. Please try again.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-based poker game vs AI")
    parser.add_argument('--model', type=str, default='poker_bot_final_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create and run game
    game = TextPokerGame(model_path=args.model)
    game.run()


if __name__ == '__main__':
    main()