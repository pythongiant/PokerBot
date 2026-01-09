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
| Win Rate vs Random | 0.520 |
| Average Reward | 0.500 |
| Reward Std Dev | 1.847 |

**Exploitability:** Not computed

## Belief State Analysis

### Belief State Geometry

| Statistic | Value |
|-----------|-------|
| Samples | 32 |
| Mean Value | -0.002 |
| Value Std Dev | 0.002 |

**Belief Dimension Statistics (first 5):**

| Dim | Mean | Std |
|-----|------|-----|
| 0 | -0.842 | 0.038 |
| 1 | -1.402 | 0.076 |
| 2 | -1.232 | 0.057 |
| 3 | -0.463 | 0.063 |
| 4 | 0.020 | 0.030 |

*See belief_geometry.png for 2D projection visualizations*

## Attention Analysis

### Attention Patterns

**Attention Entropy by Layer:**

| Layer | Mean Entropy | Std Entropy |
|-------|-------------|-------------|
| layer_0 | 0.875 | 0.686 |
| layer_1 | 0.881 | 0.690 |
| layer_2 | 0.881 | 0.691 |

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

- **Low win rate vs random**: Focus on basic strategy learning

