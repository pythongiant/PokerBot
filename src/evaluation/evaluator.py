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
import random
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import json
import logging

from src.model import PokerTransformerAgent, BeliefStateGeometry
from src.environment import KuhnPoker, Action, ObservableState
from .visualizer import BeliefStateVisualizer
from .cfr_solver import KuhnCFRSolver
from .report_generator import EvaluationReportGenerator


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
                    legal_action_indices = [a.value for a in legal_actions]

                    # Greedy action selection
                    policy = F.softmax(policy_logits[0], dim=-1).cpu().numpy()
                    legal_mask = np.zeros(4)
                    legal_mask[legal_action_indices] = 1.0
                    policy = policy * legal_mask
                    policy = policy / (policy.sum() + 1e-8)
                    action_idx = np.argmax(policy)
                    action = Action(action_idx)
            else:
                # Random opponent
                legal_actions = env.get_legal_actions(current_player)
                action = random.choice(legal_actions)

            game_state, obs, _ = env.step(current_player, action, amount=1)
            current_player = 1 - current_player
        
        if game_state.is_terminal and game_state.payoffs is not None:
            return game_state.payoffs[player_id]
        else:
            # Game didn't reach terminal state
            return 0.0
    
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
                legal_action_indices = [a.value for a in legal_actions]

                policy = F.softmax(policy_logits[0], dim=-1).cpu().numpy()
                legal_mask = np.zeros(4)
                legal_mask[legal_action_indices] = 1.0
                policy = policy * legal_mask
                policy = policy / (policy.sum() + 1e-8)
                action_idx = np.argmax(policy)
                action = Action(action_idx)
                
                game_state, obs, _ = env.step(current_player, Action(action), amount=1)
                current_player = 1 - current_player
        
        if game_state.is_terminal and game_state.payoffs is not None:
            return game_state.payoffs[return_perspective]
        else:
            return 0.0
    
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
                    action = random.choice(legal_actions)
                    game_state, obs, _ = env.step(game_state.current_player,
                                                action, amount=1)
        
        all_beliefs = np.array(all_beliefs)
        all_values = np.array(all_values)

        belief_mean = all_beliefs.mean(axis=0)
        belief_std = all_beliefs.std(axis=0)

        # Ensure we get 1D arrays
        if belief_mean.ndim > 1:
            belief_mean = belief_mean.flatten()
        if belief_std.ndim > 1:
            belief_std = belief_std.flatten()

        results = {
            'belief_mean': belief_mean.tolist(),
            'belief_std': belief_std.tolist(),
            'belief_var': all_beliefs.var(axis=0).tolist(),
            'value_mean': float(all_values.mean()),
            'value_std': float(all_values.std()),
            'num_samples': len(all_beliefs),
        }
        
        return results
    
    def compute_exploitability(self, use_transition_model: bool = True) -> Optional[float]:
        """
        Compute exploitability via CFR baseline.

        Args:
            use_transition_model: Whether to use learned transition model for evaluation

        Returns:
            exploitability: Best-response value (lower is better, 0 is Nash)
        """
        try:
            cfr_solver = KuhnCFRSolver(max_iterations=10000)
            nash_strategy = cfr_solver.train(iterations=5000)  # Train CFR to convergence

            # Convert agent policy to CFR format
            agent_policy = self._extract_agent_policy()

            # Compute exploitability
            exploitability = cfr_solver.compute_exploitability(agent_policy)

            self.logger.info(f"Computed exploitability: {exploitability:.6f}")
            return exploitability

        except Exception as e:
            self.logger.warning(f"Exploitability computation failed: {e}")
            return None

    def _extract_agent_policy(self) -> Dict[str, List[float]]:
        """
        Extract agent's policy in CFR format.

        Returns:
            policy: Dict[infoset_key -> action_probs]
        """
        policy = {}

        # Sample games to extract policy
        env = KuhnPoker(seed=42)

        self.agent.eval()
        with torch.no_grad():
            for game_idx in range(20):  # Sample fewer games for policy extraction
                game_state, obs = env.reset()

                step_count = 0
                while not game_state.is_terminal and step_count < 10:  # Limit steps
                    current_player = game_state.current_player

                    # Get agent policy
                    belief, _ = self.agent.encode_belief([obs])
                    policy_logits = self.agent.predict_policy(belief)[0].cpu().numpy()

                    # Convert to probabilities
                    policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1).numpy()

                    # Get legal actions
                    legal_actions = env.get_legal_actions(current_player)
                    if not legal_actions:
                        break
                    legal_indices = [a.value for a in legal_actions]

                    # Mask illegal actions
                    full_policy = np.zeros(4)
                    full_policy[legal_indices] = policy_probs[legal_indices]
                    full_policy = full_policy / (full_policy.sum() + 1e-8)

                    # Create infoset key (convert card to string)
                    own_card = game_state.private_cards[current_player]
                    card_names = ['J', 'Q', 'K']
                    card_str = card_names[own_card] if isinstance(own_card, int) and 0 <= own_card < len(card_names) else 'J'
                    history = self._get_history_string(game_state.public_bets[current_player])
                    infoset_key = f"{card_str}_{history}"

                    # Store policy (only for current player)
                    if current_player == 0:  # Only evaluate player 0's policy
                        policy[infoset_key] = full_policy.tolist()

                    # Random action to continue
                    action = random.choice(legal_actions)
                    game_state, obs, _ = env.step(current_player, action, amount=1)
                    step_count += 1

        return policy

    def _get_history_string(self, bets: List) -> str:
        """Convert betting history to string format."""
        history = ""
        for action, amount in bets:
            if action == Action.CHECK:
                history += "c"
            elif action == Action.CALL:
                history += "c"
            elif action == Action.RAISE:
                history += "r"
            elif action == Action.FOLD:
                history += "f"
        return history

    def _generate_attention_heatmaps(self):
        """Generate attention heatmap visualizations."""
        from src.evaluation.visualizer import BeliefStateVisualizer

        visualizer = BeliefStateVisualizer(self.agent, self.config, self.output_dir)

        # Sample some games to get attention data
        env = KuhnPoker(seed=42)

        self.agent.eval()
        with torch.no_grad():
            for game_idx in range(3):  # Generate heatmaps for a few games
                game_state, obs = env.reset()

                step_count = 0
                while not game_state.is_terminal and step_count < 5:  # Limit steps
                    current_player = game_state.current_player

                    # Get attention data
                    outputs = self.agent([obs])
                    attention_weights = outputs['attention_weights']

                    # Generate heatmap for each layer and head
                    for layer_idx in range(len(attention_weights)):
                        for head_idx in range(attention_weights[layer_idx].shape[1]):
                            # Create action history for semantic labels
                            action_history = [(p, a, amt) for p, a, amt in obs.action_history]

                            try:
                                visualizer.plot_attention_heatmap(
                                    attention_weights[layer_idx],
                                    layer_idx=layer_idx,
                                    head_idx=head_idx,
                                    title=f"Attention Heatmap - Game {game_idx}, Step {step_count}",
                                    action_history=action_history
                                )
                            except Exception as e:
                                self.logger.warning(f"Failed to generate attention heatmap L{layer_idx} H{head_idx}: {e}")

                    # Step game
                    legal_actions = env.get_legal_actions(current_player)
                    if not legal_actions:
                        break
                    action = random.choice(legal_actions)
                    game_state, obs, _ = env.step(current_player, action, amount=1)
                    step_count += 1

    def probe_belief_attention(self, num_games: int = 5) -> Dict:
        """
        Analyze attention patterns to understand belief state learning.

        Returns:
            results: Attention statistics per layer and position
        """
        env = KuhnPoker(seed=42)

        layer_attention_stats = {}
        semantic_attention_stats = []
        belief_geometry = BeliefStateGeometry(self.agent)

        self.agent.eval()

        with torch.no_grad():
            for game_idx in range(num_games):
                game_state, obs = env.reset()
                game_history = []

                while not game_state.is_terminal:
                    outputs = self.agent([obs])  # This returns attention weights

                    attention_weights = outputs['attention_weights']

                    # Store action history for semantic analysis
                    current_history = obs.action_history.copy()

                    # Analyze attention across layers
                    for layer_idx, attn_w in enumerate(attention_weights):
                        # attn_w: (batch, heads, seq_len, seq_len)
                        if f'layer_{layer_idx}' not in layer_attention_stats:
                            layer_attention_stats[f'layer_{layer_idx}'] = []

                        # Entropy of final position's attention
                        final_attn = attn_w[0, :, -1, :].mean(dim=0)  # Average over heads
                        entropy = -(final_attn * (final_attn + 1e-10).log()).sum()
                        layer_attention_stats[f'layer_{layer_idx}'].append(entropy.item())

                    # Semantic attention analysis
                    semantic_analysis = belief_geometry.analyze_attention_semantics(
                        attention_weights, current_history
                    )
                    semantic_attention_stats.append(semantic_analysis)

                    # Step game
                    legal_actions = env.get_legal_actions(game_state.current_player)
                    action = random.choice(legal_actions)
                    game_state, obs, _ = env.step(game_state.current_player,
                                                action, amount=1)

        # Compute statistics
        results = {}
        for layer, entropies in layer_attention_stats.items():
            results[layer] = {
                'mean_entropy': float(np.mean(entropies)),
                'std_entropy': float(np.std(entropies)),
            }

        # Add semantic attention analysis
        if semantic_attention_stats:
            # Aggregate semantic statistics
            semantic_agg = {}
            for analysis in semantic_attention_stats:
                for layer, layer_data in analysis.items():
                    if layer not in semantic_agg:
                        semantic_agg[layer] = {}
                    for action, weight in layer_data['attention_by_action'].items():
                        if action not in semantic_agg[layer]:
                            semantic_agg[layer][action] = []
                        semantic_agg[layer][action].append(weight)

            # Compute means
            for layer in semantic_agg:
                results[layer]['attention_by_action'] = {}
                for action in semantic_agg[layer]:
                    weights = semantic_agg[layer][action]
                    results[layer]['attention_by_action'][action] = float(np.mean(weights))

        return results
    
    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation suite."""
        self.logger.info("Running full evaluation...")
        
        results = {
            'vs_random': self.evaluate_vs_random(num_games=100),
            'belief_analysis': self.analyze_belief_states(num_games=10),
            'attention_analysis': self.probe_belief_attention(num_games=5),
            'exploitability': self.compute_exploitability(),
        }
        
        # Save results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)

        self.logger.info(f"Evaluation complete. Results saved to {results_path}")

        # Generate attention heatmaps
        self.logger.info("Generating attention heatmaps...")
        try:
            self._generate_attention_heatmaps()
        except Exception as e:
            self.logger.warning(f"Attention heatmap generation failed: {e}")

        # Generate comprehensive report
        try:
            report_generator = EvaluationReportGenerator(self.output_dir)
            report_path = report_generator.generate_report(results)
            self.logger.info(f"Evaluation report generated at {report_path}")
        except Exception as e:
            self.logger.warning(f"Report generation failed: {e}")

        return results
