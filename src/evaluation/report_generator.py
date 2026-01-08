"""
Evaluation report generator.

Creates comprehensive markdown reports compiling metrics, visualizations, and analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging


class EvaluationReportGenerator:
    """
    Generates comprehensive evaluation reports in markdown format.

    Compiles loss curves, belief geometry, game outcome stats, and policy heatmaps.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_report(self, evaluation_results: Dict,
                       training_metrics: Optional[Dict] = None,
                       checkpoint_path: Optional[Path] = None) -> Path:
        """
        Generate comprehensive evaluation report.

        Args:
            evaluation_results: Results from PokerEvaluator.run_full_evaluation()
            training_metrics: Training metrics (loss curves, etc.)
            checkpoint_path: Path to model checkpoint

        Returns:
            report_path: Path to generated markdown report
        """
        report_path = self.output_dir / "evaluation_report.md"

        with open(report_path, 'w') as f:
            f.write("# Poker Transformer Agent Evaluation Report\n\n")

            # Header with metadata
            f.write(self._generate_header(checkpoint_path))

            # Performance Summary
            f.write("## Performance Summary\n\n")
            f.write(self._generate_performance_summary(evaluation_results))

            # Training Progress
            if training_metrics:
                f.write("## Training Progress\n\n")
                f.write(self._generate_training_section(training_metrics))

            # Belief State Analysis
            f.write("## Belief State Analysis\n\n")
            f.write(self._generate_belief_analysis(evaluation_results))

            # Attention Analysis
            f.write("## Attention Analysis\n\n")
            f.write(self._generate_attention_analysis(evaluation_results))

            # Game Outcome Statistics
            f.write("## Game Outcome Statistics\n\n")
            f.write(self._generate_game_stats(evaluation_results))

            # Visualizations
            f.write("## Visualizations\n\n")
            f.write(self._generate_visualizations_section(evaluation_results))

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write(self._generate_recommendations(evaluation_results))

        self.logger.info(f"Generated evaluation report at {report_path}")
        return report_path

    def _generate_header(self, checkpoint_path: Optional[Path]) -> str:
        """Generate report header with metadata."""
        header = f"**Report Generated:** {Path(__file__).stat().st_mtime}\n\n"

        if checkpoint_path and checkpoint_path.exists():
            # Could load checkpoint metadata here
            header += f"**Model Checkpoint:** {checkpoint_path.name}\n\n"

        header += "**Evaluation Metrics:**\n"
        header += "- Self-play performance vs random opponent\n"
        header += "- Exploitability (CFR-based)\n"
        header += "- Belief state geometry analysis\n"
        header += "- Attention pattern analysis\n\n"

        return header

    def _generate_performance_summary(self, results: Dict) -> str:
        """Generate performance summary section."""
        summary = "### Key Metrics\n\n"

        # Self-play performance
        if 'vs_random' in results:
            vs_random = results['vs_random']
            summary += "| Metric | Value |\n"
            summary += "|--------|-------|\n"
            summary += f"| Win Rate vs Random | {vs_random['win_rate']:.3f} |\n"
            summary += f"| Average Reward | {vs_random['avg_reward']:.3f} |\n"
            summary += f"| Reward Std Dev | {vs_random['std_reward']:.3f} |\n"
            summary += "\n"

        # Exploitability
        if 'exploitability' in results and results['exploitability'] is not None:
            exp = results['exploitability']
            summary += f"**Exploitability:** {exp:.6f}\n\n"
            if exp < 0.01:
                summary += "*Very strong performance - near Nash equilibrium*\n\n"
            elif exp < 0.1:
                summary += "*Good performance - exploitable but reasonable*\n\n"
            else:
                summary += "*Significant exploitability - needs improvement*\n\n"
        else:
            summary += "**Exploitability:** Not computed\n\n"

        return summary

    def _generate_training_section(self, metrics: Dict) -> str:
        """Generate training progress section."""
        section = "### Loss Curves\n\n"

        # Check which metrics are available
        available_metrics = []
        if 'game_reward' in metrics and metrics['game_reward']:
            available_metrics.append('game_reward')
        if 'policy_loss' in metrics and metrics['policy_loss']:
            available_metrics.append('policy_loss')
        if 'value_loss' in metrics and metrics['value_loss']:
            available_metrics.append('value_loss')
        if 'transition_loss' in metrics and metrics['transition_loss']:
            available_metrics.append('transition_loss')

        if available_metrics:
            section += "| Iteration | " + " | ".join([m.replace('_', ' ').title() for m in available_metrics]) + " |\n"
            section += "|" + "|".join(["---"] * (len(available_metrics) + 1)) + "|\n"

            max_iter = max(len(metrics[m]) for m in available_metrics)
            for i in range(max_iter):
                row = f"| {i} |"
                for metric in available_metrics:
                    if i < len(metrics[metric]):
                        value = metrics[metric][i]
                        if isinstance(value, (int, float)):
                            row += f" {value:.4f} |"
                        else:
                            row += f" {float(value):.4f} |"
                    else:
                        row += " - |"
                section += row + "\n"

            section += "\n*See visualizations directory for loss curve plots*\n\n"
        else:
            section += "*No training metrics available*\n\n"

        return section

    def _generate_belief_analysis(self, results: Dict) -> str:
        """Generate belief state analysis section."""
        section = "### Belief State Geometry\n\n"

        if 'belief_analysis' in results:
            analysis = results['belief_analysis']

            section += "| Statistic | Value |\n"
            section += "|-----------|-------|\n"
            section += f"| Samples | {analysis['num_samples']} |\n"
            section += f"| Mean Value | {analysis['value_mean']:.3f} |\n"
            section += f"| Value Std Dev | {analysis['value_std']:.3f} |\n"

            # Show first few belief dimensions
            if 'belief_mean' in analysis and len(analysis['belief_mean']) > 0:
                belief_mean = analysis['belief_mean']
                belief_std = analysis['belief_std']
                section += "\n**Belief Dimension Statistics (first 5):**\n\n"
                section += "| Dim | Mean | Std |\n"
                section += "|-----|------|-----|\n"
                for i in range(min(5, len(belief_mean))):
                    mean_val = belief_mean[i] if isinstance(belief_mean[i], (int, float)) else float(belief_mean[i])
                    std_val = belief_std[i] if isinstance(belief_std[i], (int, float)) else float(belief_std[i])
                    section += f"| {i} | {mean_val:.3f} | {std_val:.3f} |\n"

            section += "\n*See belief_geometry.png for 2D projection visualizations*\n\n"
        else:
            section += "*Belief analysis not available*\n\n"

        return section

    def _generate_attention_analysis(self, results: Dict) -> str:
        """Generate attention analysis section."""
        section = "### Attention Patterns\n\n"

        if 'attention_analysis' in results:
            analysis = results['attention_analysis']

            section += "**Attention Entropy by Layer:**\n\n"
            section += "| Layer | Mean Entropy | Std Entropy |\n"
            section += "|-------|-------------|-------------|\n"

            for layer_name, stats in analysis.items():
                section += f"| {layer_name} | {stats['mean_entropy']:.3f} | {stats['std_entropy']:.3f} |\n"

            section += "\n*Higher entropy indicates more diffuse attention. Lower entropy indicates focused attention.*\n\n"
            section += "*See attention_heatmap_*.png files for detailed attention visualizations*\n\n"
        else:
            section += "*Attention analysis not available*\n\n"

        return section

    def _generate_game_stats(self, results: Dict) -> str:
        """Generate game outcome statistics section."""
        section = "### Game Outcomes\n\n"

        # This would be expanded with actual game statistics
        # For now, just reference the sample games
        section += "*Sample game visualizations available in games/ directory*\n\n"
        section += "*See sample_game_*.png files for detailed game trajectories*\n\n"

        return section

    def _generate_visualizations_section(self, results: Dict) -> str:
        """Generate visualizations section."""
        section = "### Available Visualizations\n\n"

        # Check for visualization metadata
        if 'visualizations' in results:
            viz = results['visualizations']
            if viz:
                section += "| Visualization | Description |\n"
                section += "|---------------|-------------|\n"

                viz_descriptions = {
                    'belief_projection_pca': '2D PCA projection of belief states',
                    'value_landscape': 'Value function over belief state projections',
                    'belief_evolution': 'How belief states change during games',
                    'attention_heatmap': 'Attention weights for transformer layers',
                    'training_metrics': 'Loss curves and training progress',
                    'belief_geometry': 'Belief state geometry analysis',
                    'training_summary': 'Comprehensive training dashboard'
                }

                for viz_name, viz_path in viz.items():
                    desc = viz_descriptions.get(viz_name, f'{viz_name} visualization')
                    section += f"| [{viz_name}]({viz_path}) | {desc} |\n"

                section += "\n"
            else:
                section += "*No visualizations generated*\n\n"
        else:
            section += "*Visualization metadata not available*\n\n"

        return section

    def _generate_recommendations(self, results: Dict) -> str:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        # Check exploitability
        if 'exploitability' in results and results['exploitability'] is not None:
            exp = results['exploitability']
            if exp > 0.1:
                recommendations.append("- **High exploitability detected**: Consider additional training or hyperparameter tuning")
            elif exp > 0.01:
                recommendations.append("- **Moderate exploitability**: Policy is reasonable but could be improved")
            else:
                recommendations.append("- **Low exploitability**: Policy is near optimal")

        # Check self-play performance
        if 'vs_random' in results:
            win_rate = results['vs_random']['win_rate']
            if win_rate < 0.6:
                recommendations.append("- **Low win rate vs random**: Focus on basic strategy learning")
            elif win_rate > 0.9:
                recommendations.append("- **High win rate vs random**: Good basic performance")

        # Attention analysis
        if 'attention_analysis' in results:
            # Could add attention-based recommendations
            pass

        if not recommendations:
            recommendations.append("- Continue training to further improve performance")

        section = "\n".join(recommendations) + "\n\n"

        return section