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
| Win Rate vs Random | 0.620 |
| Average Reward | 0.830 |
| Reward Std Dev | 4.077 |

**Exploitability:** 0.166701

*Significant exploitability - needs improvement*

## Belief State Analysis

### Belief State Geometry

| Statistic | Value |
|-----------|-------|
| Samples | 28 |
| Mean Value | -0.025 |
| Value Std Dev | 0.003 |

**Belief Dimension Statistics (first 5):**

| Dim | Mean | Std |
|-----|------|-----|
| 0 | 0.804 | 0.066 |
| 1 | -0.247 | 0.106 |
| 2 | -0.580 | 0.102 |
| 3 | 0.288 | 0.064 |
| 4 | 0.948 | 0.089 |

*See belief_geometry.png for 2D projection visualizations*

## Attention Analysis

### Attention Patterns

**Attention Entropy by Layer:**

| Layer | Mean Entropy | Std Entropy |
|-------|-------------|-------------|
| layer_0 | 0.345 | 0.345 |
| layer_1 | 0.346 | 0.346 |
| layer_2 | 0.346 | 0.346 |
| layer_3 | 0.346 | 0.346 |

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

