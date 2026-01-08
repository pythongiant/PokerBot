"""
Evaluation utilities for Poker Transformer Agent.

Includes:
1. Head-to-head evaluation vs baselines
2. Exploitability calculation
3. Belief state analysis and visualization
4. Policy extraction and comparison
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import json
import logging

from src.model import PokerTransformerAgent
from src.environment import KuhnPoker, Action, ObservableState


class PokerEvaluator:
    """Comprehensive evaluation suite for trained agents."""
    
    def __init__(self, agent: PokerTransformerAgent, config, 
                 output_dir: Optional[Path] = None):
        self.agent = agent
        self.config = config
        self.device = next(agent.parameters()).device
        self.output_dir = output_dir or Path("./eval_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_vs_random(self, num_games: int = 100) -> Dict[str, float]:
        """
        Evaluate agent vs random baseline.
        
        Args:
            num_games: Number of games to play
            
        Returns:
            eval_dict: Win rate, average reward, variance
        """
        env = KuhnPoker(seed=42)
        
        wins = 0
        rewards = []
        
        for game_id in range(num_games):
            reward = self._play_game_vs_random(env, player_id=0)
            rewards.append(reward)
            if reward > 0:
                wins += 1
        
        results = {
            'win_rate': wins / num_games,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'num_games': num_games,
        }
        
        self.logger.info(
            f"Agent vs Random: Win rate={results['win_rate']:.3f}, "
            f"Avg reward={results['avg_reward']:.3f}"
        )
        
        return results
    
    def _play_game_vs_random(self, env: KuhnPoker, player_id: int = 0) -> float:
        """Play one game: agent vs random opponent."""
        game_state, obs = env.reset()
        current_player = 0
        
        while not game_state.is_terminal:
            if current_player == player_id:
                # Agent's turn
                with torch.no_grad():
                    belief, _ = self.agent.encode_belief([obs])
                    policy_logits = self.agent.predict_policy(belief)
                    
                    legal_actions = env.get_legal_actions(current_player)
                    
                    # Greedy action selection
                    policy = F.softmax(policy_logits[0], dim=-1).cpu().numpy()
                    legal_mask = np.zeros(4)
                    legal_mask[legal_actions] = 1.0
                    policy = policy * legal_mask
                    policy = policy / (policy.sum() + 1e-8)
                    action = np.argmax(policy)
            else:
                # Random opponent
                legal_actions = env.get_legal_actions(current_player)
                action = np.random.choice(legal_actions)
            
            game_state, obs, _ = env.step(current_player, Action(action), amount=1)
            current_player = 1 - current_player
        
        return game_state.payoffs[player_id]
    
    def head_to_head_vs_checkpoint(self, checkpoint_path: Path, 
                                   num_games: int = 50) -> Dict[str, float]:
        """
        Evaluate current agent vs previous checkpoint.
        
        Args:
            checkpoint_path: Path to previous checkpoint
            num_games: Total games (agent plays both sides)
            
        Returns:
            eval_dict: Head-to-head statistics
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        opponent = PokerTransformerAgent(self.config).to(self.device)
        opponent.load_state_dict(checkpoint['agent_state'])
        opponent.eval()
        
        self.agent.eval()
        env = KuhnPoker(seed=42)
        
        # Play as player 0 and player 1
        p0_rewards = []
        p1_rewards = []
        
        for game_id in range(num_games // 2):
            reward = self._play_game_between_agents(env, self.agent, opponent, 0)
            p0_rewards.append(reward)
            
            reward = self._play_game_between_agents(env, opponent, self.agent, 0)
            p1_rewards.append(-reward)  # Opponent's perspective
        
        all_rewards = p0_rewards + p1_rewards
        
        results = {
            'current_win_rate': sum(1 for r in all_rewards if r > 0) / len(all_rewards),
            'current_avg_reward': np.mean(all_rewards),
            'current_std_reward': np.std(all_rewards),
            'num_games': len(all_rewards),
        }
        
        self.logger.info(
            f"Head-to-head vs checkpoint: Win rate={results['current_win_rate']:.3f}, "
            f"Avg reward={results['current_avg_reward']:.3f}"
        )
        
        return results
    
    def _play_game_between_agents(self, env: KuhnPoker, 
                                  agent1: PokerTransformerAgent,
                                  agent2: PokerTransformerAgent,
                                  return_perspective: int = 0) -> float:
        """Play game between two agents."""
        game_state, obs = env.reset()
        current_player = 0
        agent_list = [agent1, agent2]
        
        with torch.no_grad():
            while not game_state.is_terminal:
                agent = agent_list[current_player]
                
                belief, _ = agent.encode_belief([obs])
                policy_logits = agent.predict_policy(belief)
                
                legal_actions = env.get_legal_actions(current_player)
                
                policy = F.softmax(policy_logits[0], dim=-1).cpu().numpy()
                legal_mask = np.zeros(4)
                legal_mask[legal_actions] = 1.0
                policy = policy * legal_mask
                policy = policy / (policy.sum() + 1e-8)
                action = np.argmax(policy)
                
                game_state, obs, _ = env.step(current_player, Action(action), amount=1)
                current_player = 1 - current_player
        
        return game_state.payoffs[return_perspective]
    
    def analyze_belief_states(self, num_games: int = 10) -> Dict:
        """
        Analyze learned belief state geometry.
        
        Returns statistics about:
        - Distribution of values across dimensions
        - How values change with game progress
        - Variance and entropy
        """
        env = KuhnPoker(seed=42)
        all_beliefs = []
        all_values = []
        
        self.agent.eval()
        
        with torch.no_grad():
            for _ in range(num_games):
                game_state, obs = env.reset()
                
                while not game_state.is_terminal:
                    belief, _ = self.agent.encode_belief([obs])
                    value = self.agent.predict_value(belief)
                    
                    all_beliefs.append(belief.cpu().numpy())
                    all_values.append(value.item())
                    
                    legal_actions = env.get_legal_actions(game_state.current_player)
                    action = np.random.choice(legal_actions)
                    game_state, obs, _ = env.step(game_state.current_player, 
                                                   Action(action), amount=1)
        
        all_beliefs = np.array(all_beliefs)
        all_values = np.array(all_values)
        
        results = {
            'belief_mean': all_beliefs.mean(axis=0).tolist(),
            'belief_std': all_beliefs.std(axis=0).tolist(),
            'belief_var': all_beliefs.var(axis=0).tolist(),
            'value_mean': float(all_values.mean()),
            'value_std': float(all_values.std()),
            'num_samples': len(all_beliefs),
        }
        
        return results
    
    def compute_exploitability(self) -> Optional[float]:
        """
        Compute exploitability via CFR baseline (if available).
        
        This requires solving Kuhn poker exactly with CFR.
        For now, returns None - would integrate with nashpy or gambit.
        """
        # TODO: Integrate with game theory library
        # This would compute agent's best response value
        return None
    
    def probe_belief_attention(self, num_games: int = 5) -> Dict:
        """
        Analyze attention patterns to understand belief state learning.
        
        Returns:
            results: Attention statistics per layer and position
        """
        env = KuhnPoker(seed=42)
        
        layer_attention_stats = {}
        
        self.agent.eval()
        
        with torch.no_grad():
            for _ in range(num_games):
                game_state, obs = env.reset()
                
                while not game_state.is_terminal:
                    outputs = self.agent(
                        [obs]
                    )  # This returns attention weights
                    
                    attention_weights = outputs['attention_weights']
                    
                    # Analyze attention across layers
                    for layer_idx, attn_w in enumerate(attention_weights):
                        # attn_w: (batch, heads, seq_len, seq_len)
                        if f'layer_{layer_idx}' not in layer_attention_stats:
                            layer_attention_stats[f'layer_{layer_idx}'] = []
                        
                        # Entropy of final position's attention
                        final_attn = attn_w[0, :, -1, :].mean(dim=0)  # Average over heads
                        entropy = -(final_attn * (final_attn + 1e-10).log()).sum()
                        layer_attention_stats[f'layer_{layer_idx}'].append(entropy.item())
                    
                    # Step game
                    legal_actions = env.get_legal_actions(game_state.current_player)
                    action = np.random.choice(legal_actions)
                    game_state, obs, _ = env.step(game_state.current_player,
                                                   Action(action), amount=1)
        
        # Compute statistics
        results = {}
        for layer, entropies in layer_attention_stats.items():
            results[layer] = {
                'mean_entropy': float(np.mean(entropies)),
                'std_entropy': float(np.std(entropies)),
            }
        
        return results
    
    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation suite."""
        self.logger.info("Running full evaluation...")
        
        results = {
            'vs_random': self.evaluate_vs_random(num_games=100),
            'belief_analysis': self.analyze_belief_states(num_games=10),
            'attention_analysis': self.probe_belief_attention(num_games=5),
        }
        
        # Save results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        self.logger.info(f"Evaluation complete. Results saved to {results_path}")
        
        return results
