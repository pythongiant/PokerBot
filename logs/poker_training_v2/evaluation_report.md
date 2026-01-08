# Poker Transformer Agent Evaluation Report

**Report Generated:** 1767862493.9433246

**Model Checkpoint:** checkpoint_iter0.pt

**Evaluation Metrics:**
- Self-play performance vs random opponent
- Exploitability (CFR-based)
- Belief state geometry analysis
- Attention pattern analysis

## Performance Summary

### Key Metrics

| Metric | Value |
|--------|-------|
| Win Rate vs Random | 0.730 |
| Average Reward | 1.160 |
| Reward Std Dev | 3.085 |

**Exploitability:** Not computed

## Belief State Analysis

### Belief State Geometry

| Statistic | Value |
|-----------|-------|
| Samples | 22 |
| Mean Value | 0.027 |
| Value Std Dev | 0.005 |

**Belief Dimension Statistics (first 5):**

| Dim | Mean | Std |
|-----|------|-----|
| 0 | -0.637 | 0.145 |
| 1 | 0.893 | 0.174 |
| 2 | 0.597 | 0.146 |
| 3 | 1.391 | 0.073 |
| 4 | 0.976 | 0.074 |

*See belief_geometry.png for 2D projection visualizations*

## Attention Analysis

### Attention Patterns

**Attention Entropy by Layer:**

| Layer | Mean Entropy | Std Entropy |
|-------|-------------|-------------|
| layer_0 | 0.345 | 0.345 |
| layer_1 | 0.346 | 0.346 |
| layer_2 | 0.347 | 0.347 |
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

- Continue training to further improve performance

