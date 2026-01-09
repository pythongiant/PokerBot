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
| Win Rate vs Random | 0.740 |
| Average Reward | 1.080 |
| Reward Std Dev | 2.838 |

**Exploitability:** Not computed

## Belief State Analysis

### Belief State Geometry

| Statistic | Value |
|-----------|-------|
| Samples | 38 |
| Mean Value | 0.092 |
| Value Std Dev | 0.002 |

**Belief Dimension Statistics (first 5):**

| Dim | Mean | Std |
|-----|------|-----|
| 0 | -0.336 | 0.055 |
| 1 | -0.651 | 0.049 |
| 2 | -2.134 | 0.059 |
| 3 | -0.991 | 0.051 |
| 4 | 0.163 | 0.064 |

*See belief_geometry.png for 2D projection visualizations*

## Attention Analysis

### Attention Patterns

**Attention Entropy by Layer:**

| Layer | Mean Entropy | Std Entropy |
|-------|-------------|-------------|
| layer_0 | 0.602 | 0.505 |
| layer_1 | 0.602 | 0.505 |
| layer_2 | 0.602 | 0.505 |

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

- Continue training to further improve performance

