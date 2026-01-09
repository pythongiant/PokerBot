# Poker Transformer Agent Evaluation Report

**Report Generated:** 1767862493.9433246

**Evaluation Metrics:**
- Self-play performance vs random opponent
- Exploitability (CFR-based)
- Belief state geometry analysis
- Attention pattern analysis

## Performance Summary

### Key Metrics

| Metric | Value |
|--------|-------|
| Win Rate vs Random | 0.500 |
| Average Reward | -0.030 |
| Reward Std Dev | 3.835 |

**Exploitability:** 0.166701

*Significant exploitability - needs improvement*

## Belief State Analysis

### Belief State Geometry

| Statistic | Value |
|-----------|-------|
| Samples | 28 |
| Mean Value | 0.078 |
| Value Std Dev | 0.000 |

**Belief Dimension Statistics (first 5):**

| Dim | Mean | Std |
|-----|------|-----|
| 0 | -0.827 | 0.016 |
| 1 | -0.893 | 0.019 |
| 2 | 2.307 | 0.015 |
| 3 | -0.989 | 0.008 |
| 4 | -1.614 | 0.012 |

*See belief_geometry.png for 2D projection visualizations*

## Attention Analysis

### Attention Patterns

**Attention Entropy by Layer:**

| Layer | Mean Entropy | Std Entropy |
|-------|-------------|-------------|
| layer_0 | 0.520 | 0.439 |
| layer_1 | 0.520 | 0.439 |
| layer_2 | 0.520 | 0.439 |
| layer_3 | 0.520 | 0.439 |

*Higher entropy indicates more diffuse attention. Lower entropy indicates focused attention.*

*See attention_heatmap_*.png files for detailed attention visualizations*

## Game Outcome Statistics

### Game Outcomes

*Sample game visualizations available in games/ directory*

*See sample_game_*.png files for detailed game trajectories*

## Visualizations

### Available Visualizations

*Visualization metadata not available*

## Recommendations

- **High exploitability detected**: Consider additional training or hyperparameter tuning
- **Low win rate vs random**: Focus on basic strategy learning

