"""
Belief state visualization utilities.

Provides tools for interpreting learned representations:
- Belief state projections (t-SNE, UMAP)
- Attention heatmaps
- Value landscape plots
- Training metric dashboards
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

# Try to import optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.model import PokerTransformerAgent
from src.environment import KuhnPoker, Action


class BeliefStateVisualizer:
    """
    Visualize belief state geometry and training progress.
    
    Requires matplotlib and optional sklearn for t-SNE.
    """
    
    def __init__(self, agent: PokerTransformerAgent, config, 
                 output_dir: Optional[Path] = None):
        self.agent = agent
        self.config = config
        self.device = next(agent.parameters()).device
        self.output_dir = output_dir or Path("./visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed. Visualizations will be skipped.")
    
    def plot_belief_projection(self, beliefs: np.ndarray, 
                              actual_outcomes: Optional[np.ndarray] = None,
                              method: str = 'pca',
                              title: str = "Belief State Projection") -> Optional[Path]:
        """
        Project high-dimensional belief states to 2D for visualization.
        
        Args:
            beliefs: (n_samples, latent_dim) belief states
            actual_outcomes: (n_samples,) actual game outcomes for coloring
            method: 'pca' or 'tsne' (tsne requires sklearn)
            title: Plot title
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if method == 'pca':
            reducer = PCA(n_components=2) if HAS_SKLEARN else self._simple_pca
            projected = reducer.fit_transform(beliefs)
            method_name = "PCA"
        elif method == 'tsne':
            if not HAS_SKLEARN:
                print("sklearn required for t-SNE. Falling back to PCA.")
                return self.plot_belief_projection(beliefs, actual_outcomes, 'pca', title)
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            projected = reducer.fit_transform(beliefs)
            method_name = "t-SNE"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if actual_outcomes is not None:
            # Color by outcome
            scatter = ax.scatter(projected[:, 0], projected[:, 1], 
                               c=actual_outcomes, cmap='RdYlGn', 
                               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Outcome')
        else:
            ax.scatter(projected[:, 0], projected[:, 1], 
                      s=50, alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Save
        path = self.output_dir / f"belief_projection_{method}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path

    def plot_exploitability_heatmap(self, exploitability_over_time: List[float],
                                   iterations: List[int],
                                   title: str = "Exploitability Over Training") -> Optional[Path]:
        """
        Plot exploitability metrics over training iterations as a heatmap.

        Args:
            exploitability_over_time: List of exploitability values
            iterations: Corresponding iteration numbers
            title: Plot title

        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create heatmap data (exploitability vs iteration)
        # For heatmap, we can show exploitability as a 1D heatmap or line plot
        ax.plot(iterations, exploitability_over_time, 'r-', linewidth=2, marker='o', markersize=4)

        # Add reference line at 0 (Nash equilibrium)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Nash Equilibrium')

        # Fill area above zero (exploitable)
        ax.fill_between(iterations,
                       [max(0, x) for x in exploitability_over_time],
                       0, color='red', alpha=0.3, label='Exploitable Region')

        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Exploitability (Best Response Value)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Save
        path = self.output_dir / "exploitability_over_time.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def _simple_pca(self, X: np.ndarray, n_components: int = 2) -> 'SimplePCA':
        """Simple PCA fallback if sklearn not available."""
        class SimplePCA:
            def fit_transform(self, X):
                # Center data
                X_centered = X - X.mean(axis=0)
                
                # Compute SVD
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                
                # Project
                return U[:, :n_components] * S[:n_components]
        
        return SimplePCA().fit_transform(X)
    
    def plot_attention_heatmap(self, attention_weights: torch.Tensor,
                                layer_idx: int = -1,
                                head_idx: int = 0,
                                title: str = "Attention Heatmap",
                                action_history: Optional[List] = None,
                                opponent_card: Optional[int] = None) -> Optional[Path]:
        """
        Visualize attention weights as heatmap with semantic overlays.

        Args:
            attention_weights: (batch, heads, seq_len, seq_len)
            layer_idx: Which layer to visualize
            head_idx: Which head to visualize
            title: Plot title
            action_history: List of (player, action, amount) tuples for semantic labels
            opponent_card: Opponent's card for strategy correlation

        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None

        # Extract attention matrix
        attn = attention_weights[0, head_idx, :, :].cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(attn, cmap='hot', aspect='auto')

        # Add semantic labels if action history provided
        if action_history:
            seq_len = attn.shape[0]
            labels = self._generate_attention_labels(action_history, seq_len)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)

            # Highlight opponent actions
            opponent_positions = [i for i, (player, _, _) in enumerate(action_history) if player == 1]
            for pos in opponent_positions:
                if pos < seq_len:
                    ax.add_patch(plt.Rectangle((pos-0.5, -0.5), 1, seq_len,
                                             fill=False, edgecolor='blue', linewidth=2, alpha=0.7))

        ax.set_xlabel('Key Position (What is attended to)')
        ax.set_ylabel('Query Position (Current focus)')
        ax.set_title(f"{title} (Layer {layer_idx}, Head {head_idx})")

        plt.colorbar(im, ax=ax, label='Attention Weight')

        # Add legend for semantic overlays
        if action_history:
            legend_elements = [
                plt.Line2D([0], [0], color='blue', linewidth=2, label='Opponent Actions')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        # Save
        path = self.output_dir / f"attention_heatmap_L{layer_idx}_H{head_idx}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def _generate_attention_labels(self, action_history: List, seq_len: int) -> List[str]:
        """Generate semantic labels for attention positions."""
        labels = []

        # Position 0 is usually own card
        labels.append("Own Card")

        # Subsequent positions are action history
        for i, (player, action, amount) in enumerate(action_history):
            if i + 1 < seq_len:  # +1 because position 0 is card
                player_name = "Opp" if player == 1 else "Self"
                action_name = action.name
                labels.append(f"{player_name} {action_name}")

        # Pad if needed
        while len(labels) < seq_len:
            labels.append(f"Pos {len(labels)}")

        return labels[:seq_len]

    def plot_attention_strategy_correlation(self,
                                           attention_weights: torch.Tensor,
                                           opponent_card: int,
                                           layer_idx: int = -1,
                                           title: str = "Attention vs Opponent Strategy") -> Optional[Path]:
        """
        Plot attention correlation with opponent card strength.

        Shows how attention to opponent actions correlates with actual opponent strategy.
        """
        if not HAS_MATPLOTLIB:
            return None

        # For Kuhn poker, correlate attention to opponent actions with card strength
        # Higher attention to opponent bets when opponent has strong cards

        fig, ax = plt.subplots(figsize=(8, 6))

        # Extract attention to opponent positions
        attn = attention_weights[0, :, -1, :].mean(dim=0).cpu().detach().numpy()  # Average over heads

        # Mock correlation analysis (would need actual game data)
        # In real implementation, correlate with actual opponent betting patterns
        card_names = ['J', 'Q', 'K']
        opponent_card_name = card_names[opponent_card]

        ax.bar(range(len(attn)), attn, alpha=0.7)
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'{title} (Opponent: {opponent_card_name})')
        ax.grid(True, alpha=0.3)

        # Save
        path = self.output_dir / f"attention_strategy_corr_L{layer_idx}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path
    
    def plot_value_landscape(self, beliefs: np.ndarray, values: np.ndarray,
                             title: str = "Value Function Landscape") -> Optional[Path]:
        """
        Plot value function over belief states (using 2D projection).
        
        Args:
            beliefs: (n_samples, latent_dim)
            values: (n_samples,)
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            return None
        
        # Project beliefs to 2D
        pca = PCA(n_components=2)
        beliefs_2d = pca.fit_transform(beliefs)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(beliefs_2d[:, 0], beliefs_2d[:, 1], 
                            c=values, cmap='coolwarm', s=50, alpha=0.7,
                            edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title(title)
        
        cbar = plt.colorbar(scatter, ax=ax, label='Value Estimate')
        
        # Save
        path = self.output_dir / "value_landscape.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_training_metrics(self, metrics: Dict[str, List[float]],
                             title: str = "Training Progress") -> Optional[Path]:
        """
        Plot training metrics over iterations.
        
        Args:
            metrics: Dict with keys like 'game_reward', 'policy_loss', 'value_loss'
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Create subplots for different metrics
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Rewards
        if 'game_reward' in metrics and metrics['game_reward']:
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(metrics['game_reward'], 'b-', linewidth=2)
            ax1.set_ylabel('Average Reward')
            ax1.set_xlabel('Iteration')
            ax1.set_title('Game Reward')
            ax1.grid(True, alpha=0.3)
        
        # Policy Loss
        if 'policy_loss' in metrics and metrics['policy_loss']:
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(metrics['policy_loss'], 'r-', linewidth=2)
            ax2.set_ylabel('KL Divergence')
            ax2.set_xlabel('Iteration')
            ax2.set_title('Policy Loss')
            ax2.grid(True, alpha=0.3)
        
        # Value Loss
        if 'value_loss' in metrics and metrics['value_loss']:
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(metrics['value_loss'], 'g-', linewidth=2)
            ax3.set_ylabel('MSE')
            ax3.set_xlabel('Iteration')
            ax3.set_title('Value Loss')
            ax3.grid(True, alpha=0.3)
        
        # Combined view
        ax4 = fig.add_subplot(gs[1, 1])
        if 'game_reward' in metrics and metrics['game_reward']:
            ax4_2 = ax4.twinx()
            line1 = ax4.plot(metrics['game_reward'], 'b-', linewidth=2, label='Reward')
            ax4.set_ylabel('Reward', color='b')
            ax4.tick_params(axis='y', labelcolor='b')
            
            if 'policy_loss' in metrics and metrics['policy_loss']:
                line2 = ax4_2.plot(metrics['policy_loss'], 'r-', linewidth=2, label='Policy Loss')
                ax4_2.set_ylabel('Policy Loss', color='r')
                ax4_2.tick_params(axis='y', labelcolor='r')
        
        ax4.set_xlabel('Iteration')
        ax4.set_title('Training Overview')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save
        path = self.output_dir / "training_metrics.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_belief_evolution(self, game_beliefs: List[np.ndarray],
                             game_outcomes: np.ndarray,
                             num_games: int = 5,
                             title: str = "Belief Evolution During Games") -> Optional[Path]:
        """
        Plot how belief states evolve during a game.
        
        Args:
            game_beliefs: List[List[belief_vectors]] for multiple games
            game_outcomes: (num_games,) outcomes
            num_games: Number of games to plot
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB or not HAS_SKLEARN:
            return None
        
        # Flatten all beliefs for projection
        all_beliefs = np.concatenate(game_beliefs)
        pca = PCA(n_components=2)
        all_beliefs_2d = pca.fit_transform(all_beliefs)
        
        fig, axes = plt.subplots(1, num_games, figsize=(5*num_games, 5))
        if num_games == 1:
            axes = [axes]
        
        # Plot each game's belief trajectory
        idx = 0
        for game_idx in range(min(num_games, len(game_beliefs))):
            ax = axes[game_idx]
            beliefs_2d = all_beliefs_2d[idx:idx+len(game_beliefs[game_idx])]
            
            # Plot trajectory
            ax.plot(beliefs_2d[:, 0], beliefs_2d[:, 1], 'b-', linewidth=2, alpha=0.7)
            ax.scatter(beliefs_2d[:, 0], beliefs_2d[:, 1], s=30, c='blue', alpha=0.5)
            
            # Highlight start and end
            ax.scatter(beliefs_2d[0, 0], beliefs_2d[0, 1], s=200, c='green', 
                      marker='o', edgecolors='black', linewidth=2, label='Start')
            ax.scatter(beliefs_2d[-1, 0], beliefs_2d[-1, 1], s=200, c='red', 
                      marker='s', edgecolors='black', linewidth=2, label='End')
            
            outcome = game_outcomes[game_idx]
            ax.set_title(f'Game {game_idx+1} (Outcome: {outcome:.1f})')
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            idx += len(game_beliefs[game_idx])
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        # Save
        path = self.output_dir / "belief_evolution.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def generate_belief_report(self, num_games: int = 20,
                              output_file: Optional[Path] = None) -> Dict:
        """
        Generate comprehensive belief visualization report.
        
        Args:
            num_games: Number of games to analyze
            output_file: Optional file to save JSON report
            
        Returns:
            Dictionary with visualization metadata
        """
        env = KuhnPoker(seed=42)
        
        all_beliefs = []
        all_values = []
        all_outcomes = []
        game_beliefs_list = []
        
        self.agent.eval()
        
        with torch.no_grad():
            for game_id in range(num_games):
                game_state, obs = env.reset()
                game_beliefs = []
                
                while not game_state.is_terminal:
                    # Encode belief
                    belief, attn_info = self.agent.encode_belief([obs])
                    belief_np = belief.cpu().numpy()[0]
                    game_beliefs.append(belief_np)
                    all_beliefs.append(belief_np)
                    
                    # Get value
                    value = self.agent.predict_value(belief)[0, 0].item()
                    all_values.append(value)
                    
                    # Step
                    legal_actions = env.get_legal_actions(game_state.current_player)
                    action = np.random.choice(legal_actions)
                    game_state, obs, _ = env.step(game_state.current_player, Action(action))
                
                all_outcomes.append(game_state.payoffs[0])
                game_beliefs_list.append(np.array(game_beliefs))
        
        all_beliefs = np.array(all_beliefs)
        all_values = np.array(all_values)
        all_outcomes = np.array(all_outcomes)
        
        # Generate visualizations
        report = {
            'num_games': num_games,
            'num_samples': len(all_beliefs),
            'visualizations': {},
        }
        
        # Belief projection
        path = self.plot_belief_projection(all_beliefs, all_outcomes, 'pca')
        if path:
            report['visualizations']['belief_projection_pca'] = str(path)
        
        # Value landscape
        path = self.plot_value_landscape(all_beliefs, all_values)
        if path:
            report['visualizations']['value_landscape'] = str(path)
        
        # Belief evolution
        path = self.plot_belief_evolution(game_beliefs_list, all_outcomes, num_games=5)
        if path:
            report['visualizations']['belief_evolution'] = str(path)

        # Exploitability heatmap (if data available)
        # This would be called separately when training metrics are available
        
        # Statistics
        report['statistics'] = {
            'belief_mean': all_beliefs.mean(axis=0).tolist()[:5],  # First 5 dims
            'belief_std': all_beliefs.std(axis=0).tolist()[:5],
            'value_mean': float(all_values.mean()),
            'value_std': float(all_values.std()),
            'outcome_mean': float(all_outcomes.mean()),
        }
        
        # Save report
        if output_file is None:
            output_file = self.output_dir / "belief_report.json"
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def visualize_training_summary(log_dir: Path, output_dir: Optional[Path] = None):
    """
    Create comprehensive visualization report from training logs.
    
    Args:
        log_dir: Directory with metrics.json
        output_dir: Where to save visualizations
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualizations")
        return
    
    output_dir = output_dir or log_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    metrics_file = log_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Plot metrics
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Game reward
    if 'game_reward' in metrics:
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(metrics['game_reward'], 'b-', linewidth=2)
        ax.fill_between(range(len(metrics['game_reward'])), metrics['game_reward'], alpha=0.3)
        ax.set_ylabel('Average Reward')
        ax.set_xlabel('Iteration')
        ax.set_title('Game Reward Over Time')
        ax.grid(True, alpha=0.3)
    
    # Policy loss
    if 'policy_loss' in metrics:
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(metrics['policy_loss'], 'r-', linewidth=2)
        ax.set_ylabel('KL Loss')
        ax.set_xlabel('Iteration')
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)
    
    # Value loss
    if 'value_loss' in metrics:
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(metrics['value_loss'], 'g-', linewidth=2)
        ax.set_ylabel('Value Loss')
        ax.set_xlabel('Iteration')
        ax.set_title('Value Loss')
        ax.grid(True, alpha=0.3)
    
    # Transition loss
    if 'transition_loss' in metrics:
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(metrics['transition_loss'], 'orange', linewidth=2)
        ax.set_ylabel('Transition Loss')
        ax.set_xlabel('Iteration')
        ax.set_title('Transition Model Loss')
        ax.grid(True, alpha=0.3)
    
    # Combined view
    ax = fig.add_subplot(gs[1, 1:])
    if 'game_reward' in metrics:
        ax.plot(metrics['game_reward'], 'b-', linewidth=2, label='Reward', alpha=0.7)
    if 'policy_loss' in metrics:
        ax.plot(metrics['policy_loss'], 'r-', linewidth=2, label='Policy Loss', alpha=0.7)
    if 'value_loss' in metrics:
        ax.plot(metrics['value_loss'], 'g-', linewidth=2, label='Value Loss', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('All Metrics Combined')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Training Summary: {log_dir.name}', fontsize=16)
    plt.tight_layout()
    
    # Save
    path = output_dir / "training_summary.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization to {path}")
