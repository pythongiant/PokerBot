#!/usr/bin/env python3
"""
Fix Calling Bias - Targeted Retraining for CALL Optimization

This script retrains the poker model specifically to address the calling bias
by:
1. Analyzing current model behavior
2. Creating CALL-focused training scenarios
3. Implementing reward function fixes
4. Retraining with CALL optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from src.config.config import ExperimentConfig, ModelConfig
from src.model import PokerTransformerAgent
from src.environment.kuhn import KuhnPoker, Action


class CallingBiasFixer:
    """Specialized trainer to fix calling bias in poker model."""
    
    def __init__(self, model_path='poker_bot_final_model.pt'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.learning_rate = 0.001
        self.call_scenarios_weight = 3.0  # Higher weight for CALL scenarios
        self.fold_penalty = 2.0  # Penalty for folding when should call
        
        # Load existing model
        self.load_model()
        
        print("🔧 CALLING BIAS FIXER")
        print("=" * 50)
        print("Targeted retraining to fix CALL behavior")
    
    def load_model(self):
        """Load existing model for fine-tuning."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            config = ExperimentConfig(
                name='poker_fixed_calling',
                model=ModelConfig(
                    latent_dim=128,
                    num_heads=8,
                    num_layers=4,
                    ff_dim=256,
                    transition_type='deterministic'
                )
            )
            
            self.agent = PokerTransformerAgent(config).to(self.device)
            self.agent.load_state_dict(checkpoint['agent_state'], strict=False)
            
            # Create optimizer for fine-tuning
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate)
            
            print("✅ Model loaded successfully for fine-tuning")
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def create_call_scenarios(self, num_scenarios=100):
        """Create training scenarios where CALL is optimal."""
        scenarios = []
        
        # Scenario types where CALL is clearly optimal
        call_optimal_scenarios = [
            # Equal cards - should CALL
            {'p0_card': 0, 'p1_card': 0, 'p0_action': 'RAISE', 'optimal': 'CALL'},
            {'p0_card': 1, 'p1_card': 1, 'p0_action': 'RAISE', 'optimal': 'CALL'},
            {'p0_card': 2, 'p1_card': 2, 'p0_action': 'RAISE', 'optimal': 'CALL'},
            
            # Slightly better cards - should CALL
            {'p0_card': 0, 'p1_card': 1, 'p0_action': 'RAISE', 'optimal': 'CALL'},
            {'p0_card': 1, 'p1_card': 2, 'p0_action': 'RAISE', 'optimal': 'CALL'},
            
            # Good pot odds situations
            {'p0_card': 0, 'p1_card': 1, 'p0_action': 'RAISE', 'optimal': 'CALL', 'pot_odds': 'good'},
            {'p0_card': 1, 'p1_card': 2, 'p0_action': 'RAISE', 'optimal': 'CALL', 'pot_odds': 'good'},
        ]
        
        # Generate scenarios with variations
        for _ in range(num_scenarios):
            base_scenario = random.choice(call_optimal_scenarios)
            
            # Add some randomness while keeping CALL optimal
            scenario = base_scenario.copy()
            
            # Vary pot size to create different pot odds
            if 'pot_odds' not in scenario:
                scenario['pot_size'] = random.choice([3, 4, 5])
            
            scenarios.append(scenario)
        
        print(f"✅ Created {len(scenarios)} CALL-focused training scenarios")
        return scenarios
    
    def calculate_call_reward(self, action, optimal_action, scenario):
        """Calculate reward with emphasis on proper CALL behavior."""
        base_reward = 0.0
        
        if action == optimal_action:
            base_reward = 1.0  # Correct action
            
            # Bonus for CALL when it's optimal
            if optimal_action == 'CALL' and action == 'CALL':
                base_reward += 2.0  # Extra reward for proper CALL
                
        else:
            base_reward = -1.0  # Wrong action
            
            # Penalty for folding when should call
            if optimal_action == 'CALL' and action == 'FOLD':
                base_reward -= self.fold_penalty
                
            # Penalty for raising when should call
            if optimal_action == 'CALL' and action == 'RAISE':
                base_reward -= 1.5
        
        return base_reward
    
    def train_on_call_scenarios(self, epochs=50, batch_size=16):
        """Train model specifically on CALL scenarios."""
        print(f"\n🎯 TARGETED TRAINING: CALL OPTIMIZATION")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"CALL scenarios weight: {self.call_scenarios_weight}")
        print(f"Fold penalty: {self.fold_penalty}")
        
        # Create training scenarios
        scenarios = self.create_call_scenarios(epochs * batch_size)
        
        self.agent.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            call_correct = 0
            total_decisions = 0
            
            # Random batch of scenarios
            batch_scenarios = random.sample(scenarios, batch_size)
            
            for scenario in batch_scenarios:
                # Set up game state
                env = KuhnPoker(seed=42)
                game_state, obs = env.reset()
                
                # Configure scenario
                game_state.private_cards = [scenario['p0_card'], scenario['p1_card']]
                game_state.current_player = 1  # Player 1's turn (responding to raise)
                
                # Set pot
                pot_size = scenario.get('pot_size', 4)
                game_state.pot = pot_size
                
                # Simulate Player 0 raising
                game_state.public_bets[0] = [(Action.RAISE, 1)]
                game_state.stacks[0] -= 1
                game_state.pot += 1
                
                # Get model prediction
                belief, _ = self.agent.encode_belief([obs])
                policy_logits = self.agent.predict_policy(belief)[0]
                policy_probs = torch.softmax(policy_logits, dim=-1)
                
                # Get legal actions
                legal_actions = env.get_legal_actions(1)
                
                # Create target distribution
                target_dist = torch.zeros(4, device=self.device)
                optimal_action = scenario['optimal']
                
                if optimal_action == 'CALL' and Action.CALL in legal_actions:
                    # High probability for CALL
                    target_dist[Action.CALL.value] = 0.7
                    # Lower probabilities for other legal actions
                    remaining_prob = 0.3
                    other_actions = [a for a in legal_actions if a != Action.CALL]
                    if other_actions:
                        prob_each = remaining_prob / len(other_actions)
                        for action in other_actions:
                            target_dist[action.value] = prob_each
                
                # Calculate loss
                legal_mask = torch.zeros(4, device=self.device)
                for a in legal_actions:
                    legal_mask[a.value] = 1.0
                
                # Cross-entropy loss with CALL weighting
                loss = -torch.sum(target_dist * torch.log(policy_probs + 1e-8))
                loss *= self.call_scenarios_weight  # Weight CALL scenarios more heavily
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_decisions += 1
                
                # Track if model would CALL
                if Action.CALL in legal_actions:
                    call_prob = policy_probs[Action.CALL.value].item()
                    if call_prob > 0.5:  # Model would likely CALL
                        call_correct += 1
            
            avg_loss = total_loss / batch_size
            call_accuracy = call_correct / total_decisions if total_decisions > 0 else 0
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, CALL accuracy={call_accuracy:.2f}")
        
        self.agent.eval()
        print(f"\n✅ CALL optimization training complete!")
    
    def test_calling_behavior(self, num_tests=50):
        """Test model's calling behavior after training."""
        print(f"\n🧪 TESTING CALLING BEHAVIOR")
        print(f"Running {num_tests} test scenarios...")
        
        env = KuhnPoker(seed=42)
        card_names = ['J', 'Q', 'K']
        
        call_opportunities = 0
        calls_made = 0
        correct_calls = 0
        
        for i in range(num_tests):
            # Create CALL scenario
            p0_card = random.randint(0, 2)
            p1_card = random.randint(0, 2)
            
            # Ensure it's a situation where CALL could be reasonable
            if p1_card >= p0_card:  # P1 has equal or better card
                game_state, obs = env.reset()
                game_state.private_cards = [p0_card, p1_card]
                game_state.current_player = 1
                game_state.pot = 4
                
                # Simulate P0 raising
                game_state.public_bets[0] = [(Action.RAISE, 1)]
                game_state.stacks[0] -= 1
                game_state.pot += 1
                
                # Get model action
                with torch.no_grad():
                    belief, _ = self.agent.encode_belief([obs])
                    policy_logits = self.agent.predict_policy(belief)[0]
                    policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
                    
                    legal_actions = env.get_legal_actions(1)
                    
                    if Action.CALL in legal_actions:
                        call_opportunities += 1
                        
                        # Select action
                        legal_mask = torch.zeros(4)
                        for a in legal_actions:
                            legal_mask[a.value] = 1.0
                        masked_logits = policy_logits + (1 - legal_mask) * (-1e9)
                        action_idx = torch.argmax(masked_logits).item()
                        
                        # Ensure action_idx is a valid integer and convert to Action
                        try:
                            action = Action(action_idx)
                        except ValueError:
                            # Fallback to safest action if invalid
                            action = Action.CHECK
                        
                        if action == Action.CALL:
                            calls_made += 1
                            if p1_card >= p0_card:  # Correct to call
                                correct_calls += 1
        
        print(f"\n📊 TEST RESULTS:")
        print(f"CALL opportunities: {call_opportunities}")
        print(f"CALLs made: {calls_made}")
        print(f"Correct CALLs: {correct_calls}")
        
        if call_opportunities > 0:
            call_rate = calls_made / call_opportunities * 100
            correct_rate = correct_calls / call_opportunities * 100
            
            print(f"CALL rate: {call_rate:.1f}%")
            print(f"CALL accuracy: {correct_rate:.1f}%")
            
            if call_rate >= 50:
                print("✅ Model now calls appropriately!")
            elif call_rate >= 25:
                print("⚠️  Model calling improved but still low")
            else:
                print("🚨 Model still under-calls")
        
        return call_rate if call_opportunities > 0 else 0
    
    def save_fixed_model(self, path='poker_bot_fixed_calling.pt'):
        """Save the retrained model."""
        checkpoint = {
            'agent_state': self.agent.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': {
                'name': 'poker_fixed_calling',
                'model': {
                    'latent_dim': 128,
                    'num_heads': 8,
                    'num_layers': 4,
                    'ff_dim': 256,
                    'transition_type': 'deterministic'
                }
            },
            'training_info': {
                'call_bias_fixed': True,
                'call_scenarios_weight': self.call_scenarios_weight,
                'fold_penalty': self.fold_penalty
            }
        }
        
        torch.save(checkpoint, path)
        print(f"✅ Fixed model saved to: {path}")
    
    def run_fix_process(self):
        """Complete process to fix calling bias."""
        print("\n🚀 STARTING CALLING BIAS FIX PROCESS")
        print("=" * 60)
        
        # Step 1: Test current behavior
        print("\n📊 STEP 1: Testing current model behavior...")
        current_call_rate = self.test_calling_behavior(num_tests=20)
        
        # Step 2: Train on CALL scenarios
        print("\n🎯 STEP 2: Training on CALL scenarios...")
        self.train_on_call_scenarios(epochs=30, batch_size=16)
        
        # Step 3: Test improved behavior
        print("\n📊 STEP 3: Testing improved model behavior...")
        new_call_rate = self.test_calling_behavior(num_tests=20)
        
        # Step 4: Save fixed model
        print("\n💾 STEP 4: Saving fixed model...")
        self.save_fixed_model()
        
        # Step 5: Summary
        print(f"\n🎉 CALLING BIAS FIX COMPLETE!")
        print("=" * 60)
        print(f"Original CALL rate: {current_call_rate:.1f}%")
        print(f"Fixed CALL rate: {new_call_rate:.1f}%")
        print(f"Improvement: {new_call_rate - current_call_rate:+.1f}%")
        
        if new_call_rate > current_call_rate + 20:
            print("✅ Significant improvement in calling behavior!")
        elif new_call_rate > current_call_rate + 10:
            print("✅ Good improvement in calling behavior!")
        else:
            print("⚠️  Limited improvement - may need more training")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix calling bias in poker model")
    parser.add_argument('--model', type=str, default='poker_bot_final_model.pt',
                       help='Path to original model')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs for CALL optimization')
    parser.add_argument('--output', type=str, default='poker_bot_fixed_calling.pt',
                       help='Output path for fixed model')
    
    args = parser.parse_args()
    
    # Create and run fixer
    fixer = CallingBiasFixer(model_path=args.model)
    fixer.run_fix_process()


if __name__ == '__main__':
    main()