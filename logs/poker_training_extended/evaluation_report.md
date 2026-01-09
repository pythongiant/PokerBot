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
| Win Rate vs Random | 0.790 |
| Average Reward | 2.720 |
| Reward Std Dev | 4.133 |

**Exploitability:** 0.166701

*Significant exploitability - needs improvement*

## Belief State Analysis

### Belief State Geometry

| Statistic | Value |
|-----------|-------|
| Samples | 26 |
| Mean Value | -0.179 |
| Value Std Dev | 0.006 |

**Belief Dimension Statistics (first 5):**

| Dim | Mean | Std |
|-----|------|-----|
| 0 | 0.510 | 0.159 |
| 1 | 0.043 | 0.275 |
| 2 | 0.303 | 0.352 |
| 3 | 0.810 | 0.163 |
| 4 | -2.066 | 0.286 |

*See belief_geometry.png for 2D projection visualizations*

## Attention Analysis

### Attention Patterns

**Attention Entropy by Layer:**

| Layer | Mean Entropy | Std Entropy |
|-------|-------------|-------------|
| layer_0 | 0.581 | 0.535 |
| layer_1 | 0.576 | 0.531 |
| layer_2 | 0.581 | 0.535 |
| layer_3 | 0.581 | 0.535 |

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

