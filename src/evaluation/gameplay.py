"""
Play and visualize sample games with trained agents.

Includes:
1. Play single game with belief tracking
2. Visualize game trajectory
3. Geometric analysis of beliefs during play
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import logging

from src.model import PokerTransformerAgent
from src.environment import KuhnPoker, Action, ObservableState, GameState
from .visualizer import BeliefStateVisualizer


class GameRecorder:
    """Record and visualize a single game."""
    
    def __init__(self, agent: PokerTransformerAgent, config, 
                 output_dir: Optional[Path] = None):
        self.agent = agent
        self.config = config
        self.device = next(agent.parameters()).device
        self.output_dir = output_dir or Path("./game_records")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def play_game(self, seed: Optional[int] = None) -> Dict:
        """
        Play one complete game with the agent.
        
        Records beliefs, actions, values, and policies at each step.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            game_record: Dictionary with game history
        """
        env = KuhnPoker(seed=seed or np.random.randint(0, 10000))
        game_state, obs = env.reset()
        
        record = {
            'seed': seed,
            'actions': [],
            'observations': [],
            'beliefs': [],
            'values': [],
            'policies': [],
            'legal_actions': [],
            'rewards': [0, 0],  # [player0_reward, player1_reward]
            'game_state': None,
        }
        
        current_player = 0
        step_count = 0
        
        while not game_state.is_terminal and step_count < 100:
            record['observations'].append(self._serialize_observation(obs))
            record['legal_actions'].append([a.name for a in env.get_legal_actions(current_player)])
            
            # Get agent decision
            with torch.no_grad():
                belief, _ = self.agent.encode_belief([obs])
                value = self.agent.predict_value(belief)[0].cpu().numpy()
                policy_logits = self.agent.predict_policy(belief)[0].cpu().numpy()
                
                # Store belief and value
                record['beliefs'].append(belief[0].cpu().numpy().tolist())
                record['values'].append(float(value))
                record['policies'].append(policy_logits.tolist())
                
                # Sample action
                legal_actions = env.get_legal_actions(current_player)
                legal_action_indices = [a.value for a in legal_actions]
                
                # Mask policy
                policy = F.softmax(torch.tensor(policy_logits, device=self.device), dim=-1).cpu().numpy()
                legal_mask = np.zeros(4)
                legal_mask[legal_action_indices] = 1.0
                policy = policy * legal_mask
                policy_sum = policy.sum()
                if policy_sum > 0:
                    policy = policy / policy_sum
                else:
                    # If all masked, use uniform over legal actions
                    policy = legal_mask / legal_mask.sum()
                
                action_idx = np.random.choice(4, p=policy)
                action = Action(action_idx)
            
            record['actions'].append(action.name)
            
            # Step environment
            game_state, obs, _ = env.step(current_player, action, amount=1)
            current_player = 1 - current_player
            step_count += 1
        
        # Get final payoff
        if game_state.is_terminal:
            payoff_p0 = env.get_payoff(0)
            payoff_p1 = env.get_payoff(1)
            record['rewards'] = [float(payoff_p0), float(payoff_p1)]
        
        record['game_state'] = self._serialize_game_state(game_state)
        record['steps'] = step_count
        
        return record
    
    def _serialize_observation(self, obs: ObservableState) -> Dict:
        """Convert observation to serializable dict."""
        return {
            'own_card': int(obs.own_card) if obs.own_card is not None else None,
            'action_history': [(int(p), a.name, int(amt)) for p, a, amt in obs.action_history],
            'stacks': [int(s) for s in obs.stacks],
            'pot': int(obs.pot),
            'current_player': int(obs.current_player),
        }
    
    def _serialize_game_state(self, gs: GameState) -> Dict:
        """Convert game state to serializable dict."""
        return {
            'private_cards': [int(c) for c in gs.private_cards],
            'public_cards': [int(c) for c in gs.public_cards],
            'stacks': [int(s) for s in gs.stacks],
            'pot': int(gs.pot),
            'is_terminal': bool(gs.is_terminal),
            'payoffs': [float(p) for p in gs.payoffs] if gs.payoffs else None,
        }
    
    def visualize_game(self, game_record: Dict, name: str = 'sample_game') -> Path:
        """
        Create visualizations for the game.
        
        Generates:
        1. Belief evolution during game
        2. Value estimates over time
        3. Policy distributions per step
        4. Game summary
        
        Args:
            game_record: Game record from play_game()
            name: Name for output files
            
        Returns:
            output_dir: Path to visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            self.logger.warning("matplotlib not installed - skipping game visualization")
            return self.output_dir
        
        # Extract data
        beliefs = np.array(game_record['beliefs'])
        values = np.array(game_record['values'])
        policies = np.array(game_record['policies'])
        actions = game_record['actions']
        legal_actions_list = game_record['legal_actions']
        rewards = game_record['rewards']
        
        num_steps = len(actions)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        # 1. Value progression
        ax1 = fig.add_subplot(gs[0, :])
        steps = np.arange(num_steps + 1)
        values_extended = np.concatenate([[0], values])
        ax1.plot(steps, values_extended, 'b-o', linewidth=2, markersize=4, label='Value Estimate')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Step', fontsize=11)
        ax1.set_ylabel('Estimated Value', fontsize=11)
        ax1.set_title('Value Function Progression During Game', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Policy entropy over time
        ax2 = fig.add_subplot(gs[1, 0])
        entropies = []
        for p in policies:
            p_softmax = F.softmax(torch.tensor(p, dtype=torch.float32), dim=-1).numpy()
            entropy = -np.sum(p_softmax * np.log(p_softmax + 1e-8))
            entropies.append(entropy)
        
        ax2.plot(entropies, 'g-o', linewidth=2, markersize=4, label='Policy Entropy')
        ax2.set_xlabel('Step', fontsize=11)
        ax2.set_ylabel('Entropy (bits)', fontsize=11)
        ax2.set_title('Policy Entropy (Uncertainty)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Belief state L2 norm (magnitude)
        ax3 = fig.add_subplot(gs[1, 1])
        belief_norms = np.linalg.norm(beliefs, axis=1)
        ax3.plot(belief_norms, 'r-o', linewidth=2, markersize=4, label='||z_t||')
        ax3.set_xlabel('Step', fontsize=11)
        ax3.set_ylabel('Belief Magnitude', fontsize=11)
        ax3.set_title('Belief State Norm Evolution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Action sequence
        ax4 = fig.add_subplot(gs[2, :])
        action_names = [a for a in actions]
        action_colors = {'CHECK': 'blue', 'FOLD': 'red', 'CALL': 'green', 'RAISE': 'orange'}
        colors = [action_colors.get(a, 'gray') for a in action_names]
        
        ax4.bar(range(len(action_names)), [1]*len(action_names), color=colors, alpha=0.7)
        ax4.set_xticks(range(len(action_names)))
        ax4.set_xticklabels(action_names, rotation=0, fontsize=10)
        ax4.set_ylabel('Step', fontsize=11)
        ax4.set_title('Action Sequence During Game', fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 1.5])
        ax4.set_yticks([])
        
        # Add legend for actions
        legend_patches = [mpatches.Patch(facecolor=color, alpha=0.7, label=action)
                         for action, color in action_colors.items()]
        ax4.legend(handles=legend_patches, loc='upper right')
        
        # Overall title with game summary
        outcome_str = f"P0: {rewards[0]:+.1f} | P1: {rewards[1]:+.1f}"
        fig.suptitle(f'Sample Game Visualization - {outcome_str}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_path = self.output_dir / f'{name}_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Saved game visualization to {output_path}")
        
        return output_path
    
    def save_game_record(self, game_record: Dict, name: str = 'sample_game') -> Path:
        """Save game record to JSON."""
        output_path = self.output_dir / f'{name}_record.json'
        
        with open(output_path, 'w') as f:
            json.dump(game_record, f, indent=2)
        
        self.logger.info(f"✓ Saved game record to {output_path}")
        return output_path


def play_and_visualize_sample_game(agent: PokerTransformerAgent, config, 
                                   output_dir: Path, num_games: int = 1) -> List[Dict]:
    """
    Play multiple sample games and create visualizations.
    
    Args:
        agent: Trained PokerTransformerAgent
        config: ExperimentConfig
        output_dir: Directory for outputs
        num_games: Number of games to play
        
    Returns:
        game_records: List of game records
    """
    recorder = GameRecorder(agent, config, output_dir / 'games')
    game_records = []
    
    logger = logging.getLogger(__name__)
    
    for i in range(num_games):
        logger.info(f"Playing sample game {i+1}/{num_games}...")
        
        game_record = recorder.play_game(seed=42 + i)
        game_records.append(game_record)
        
        # Visualize
        recorder.visualize_game(game_record, name=f'sample_game_{i}')
        recorder.save_game_record(game_record, name=f'sample_game_{i}')
        
        # Log summary
        p0_reward, p1_reward = game_record['rewards']
        logger.info(f"  Game {i}: P0 {p0_reward:+.1f} | P1 {p1_reward:+.1f} | {game_record['steps']} steps")
    
    return game_records


def visualize_geometry(agent: PokerTransformerAgent, config, output_dir: Path) -> Path:
    """
    Visualize belief state geometry by playing multiple games.
    
    Creates PCA/t-SNE projection of belief states colored by game outcome.
    
    Args:
        agent: Trained PokerTransformerAgent
        config: ExperimentConfig
        output_dir: Directory for outputs
        
    Returns:
        visualization_path: Path to geometry plot
    """
    logger = logging.getLogger(__name__)
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("sklearn/matplotlib not installed - skipping geometry visualization")
        return None
    
    # Play games and collect beliefs
    all_beliefs = []
    all_outcomes = []  # 1 for P0 win, -1 for P1 win
    
    recorder = GameRecorder(agent, config, output_dir / 'games')
    
    logger.info("Collecting beliefs from 50 sample games for geometry analysis...")
    
    for i in range(50):
        game_record = recorder.play_game(seed=100 + i)
        
        # Collect beliefs and outcomes
        beliefs = np.array(game_record['beliefs'])
        outcome = game_record['rewards'][0] - game_record['rewards'][1]
        outcome_label = 1 if outcome > 0 else -1
        
        all_beliefs.append(beliefs)
        all_outcomes.extend([outcome_label] * len(beliefs))
    
    # Stack all beliefs
    all_beliefs_stacked = np.vstack(all_beliefs)
    all_outcomes = np.array(all_outcomes)
    
    # PCA projection
    pca = PCA(n_components=2)
    beliefs_pca = pca.fit_transform(all_beliefs_stacked)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA plot
    ax = axes[0]
    scatter = ax.scatter(beliefs_pca[:, 0], beliefs_pca[:, 1], 
                        c=all_outcomes, cmap='RdYlGn', alpha=0.6, s=30)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('Belief State Geometry (PCA)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Outcome (1=P0 Win, -1=P1 Win)')
    ax.grid(True, alpha=0.3)
    
    # t-SNE projection
    try:
        logger.info("Computing t-SNE projection (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        beliefs_tsne = tsne.fit_transform(all_beliefs_stacked)
        
        ax = axes[1]
        scatter = ax.scatter(beliefs_tsne[:, 0], beliefs_tsne[:, 1],
                            c=all_outcomes, cmap='RdYlGn', alpha=0.6, s=30)
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
        ax.set_title('Belief State Geometry (t-SNE)', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Outcome (1=P0 Win, -1=P1 Win)')
        ax.grid(True, alpha=0.3)
    except Exception as e:
        logger.warning(f"t-SNE failed: {e}")
        axes[1].text(0.5, 0.5, 't-SNE projection\nfailed', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    fig.suptitle('Belief State Geometry Analysis (50 games)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'belief_geometry.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved geometry visualization to {output_path}")
    
    return output_path
