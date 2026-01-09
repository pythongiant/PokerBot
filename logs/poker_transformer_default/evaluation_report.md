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
| Win Rate vs Random | 0.670 |
| Average Reward | 1.560 |
| Reward Std Dev | 4.134 |

**Exploitability:** 0.166701

*Significant exploitability - needs improvement*

## Belief State Analysis

### Belief State Geometry

| Statistic | Value |
|-----------|-------|
| Samples | 28 |
| Mean Value | -0.070 |
| Value Std Dev | 0.021 |

**Belief Dimension Statistics (first 5):**

| Dim | Mean | Std |
|-----|------|-----|
| 0 | -0.663 | 0.612 |
| 1 | -0.068 | 0.884 |
| 2 | 1.556 | 0.384 |
| 3 | 0.074 | 0.709 |
| 4 | 1.483 | 0.379 |

*See belief_geometry.png for 2D projection visualizations*

## Attention Analysis

### Attention Patterns

**Attention Entropy by Layer:**

| Layer | Mean Entropy | Std Entropy |
|-------|-------------|-------------|
| layer_0 | 0.540 | 0.472 |
| layer_1 | 0.542 | 0.473 |
| layer_2 | 0.542 | 0.473 |

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

