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
| Win Rate vs Random | 0.400 |
| Average Reward | 0.250 |
| Reward Std Dev | 1.367 |

**Exploitability:** Not computed

## Belief State Analysis

### Belief State Geometry

| Statistic | Value |
|-----------|-------|
| Samples | 38 |
| Mean Value | -0.061 |
| Value Std Dev | 0.001 |

**Belief Dimension Statistics (first 5):**

| Dim | Mean | Std |
|-----|------|-----|
| 0 | 1.119 | 0.145 |
| 1 | -1.042 | 0.222 |
| 2 | 0.738 | 0.150 |
| 3 | 1.782 | 0.141 |
| 4 | -1.990 | 0.168 |

*See belief_geometry.png for 2D projection visualizations*

## Attention Analysis

### Attention Patterns

**Attention Entropy by Layer:**

| Layer | Mean Entropy | Std Entropy |
|-------|-------------|-------------|
| layer_0 | 0.840 | 0.574 |
| layer_1 | 0.838 | 0.573 |
| layer_2 | 0.840 | 0.574 |

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

