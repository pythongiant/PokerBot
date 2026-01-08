# Belief State Visualization Guide

## Overview

The Poker Transformer now includes comprehensive visualization tools to interpret learned belief states and training progress.

## Visualizations Generated

### 1. Training Metrics Dashboard
**File**: `training_metrics.png` (automatically generated)

Shows four subplots:
- **Game Reward**: Average payoff per iteration (should increase)
- **Policy Loss**: KL divergence from targets (should decrease)
- **Value Loss**: MSE from value targets (should decrease)
- **Combined View**: Overlay of reward and loss

**What to look for**:
- ✓ Reward trending upward
- ✓ Losses trending downward
- ✓ No NaN/Inf values
- ✓ Stable convergence

### 2. Belief State Projection (PCA)
**File**: `belief_projection_pca.png`

Projects high-dimensional belief states (64-256 dims) to 2D using PCA.

- **Color**: Actual game outcomes (green=win, red=loss)
- **Position**: Similarity in belief space
- **Interpretation**: 
  - Clustered = beliefs compress well
  - Spread = beliefs encode diverse information
  - Color correlation = belief geometry relates to outcomes

**What to look for**:
- ✓ Clear separation by outcome
- ✓ Smooth gradients (no isolated clusters)
- ✓ Reasonable clustering

### 3. Value Function Landscape
**File**: `value_landscape.png`

Visualizes the learned value function over the belief space.

- **X, Y axes**: First two PCA components of belief space
- **Color**: Estimated value (blue=negative, red=positive)
- **Interpretation**:
  - Smooth gradients = well-learned value function
  - Sharp transitions = potential overfitting

**What to look for**:
- ✓ Smooth color gradients
- ✓ Values align with outcomes
- ✓ No isolated high/low peaks

### 4. Belief Evolution During Games
**File**: `belief_evolution.png`

Shows how belief states change during individual games.

- **Blue trajectory**: Path through belief space during game
- **Green circle**: Starting belief
- **Red square**: Ending belief
- **Multiple panels**: Different games

**What to look for**:
- ✓ Smooth transitions (no jumps)
- ✓ Different end positions for different outcomes
- ✓ Coherent trajectories

### 5. Attention Heatmaps
**Files**: `attention_heatmap_L*.png`

Shows which sequence positions attend to which positions.

- **X-axis**: Key positions (history positions)
- **Y-axis**: Query positions (current position)
- **Color intensity**: Attention weight

**What to look for**:
- ✓ Causal mask (upper triangle = zero)
- ✓ Diagonal focus (current position attends to recent history)
- ✓ Some positions attend to opponent actions (across players)
- ✓ Not uniform (indicates learned importance weighting)

## Generating Visualizations

### Automatic (Built-in)

Visualizations are automatically generated after training:

```bash
python main.py --num-iterations 100 --eval
# Generates all visualizations in logs/<experiment>/visualizations/
```

### Manual Generation

Generate visualizations for trained model:

```python
from pathlib import Path
from src.config import ExperimentConfig
from src.model import PokerTransformerAgent
from src.evaluation import BeliefStateVisualizer
import torch

# Load model
config = ExperimentConfig()
agent = PokerTransformerAgent(config)
checkpoint = torch.load('logs/experiment/checkpoint_iter100.pt')
agent.load_state_dict(checkpoint['agent_state'])

# Generate visualizations
visualizer = BeliefStateVisualizer(agent, config, Path('viz_output'))
report = visualizer.generate_belief_report(num_games=50)
```

### From Training Logs

Generate visualizations from metrics.json alone:

```python
from pathlib import Path
from src.evaluation import visualize_training_summary

visualize_training_summary(
    log_dir=Path('logs/my_experiment'),
    output_dir=Path('logs/my_experiment/visualizations')
)
```

## Interpreting Visualizations

### Healthy Model
- ✓ Belief projection: Separated by outcome
- ✓ Value landscape: Smooth color gradients
- ✓ Belief evolution: Smooth, coherent paths
- ✓ Attention: Causal mask respected, learned patterns
- ✓ Metrics: Consistent improvement

### Unhealthy Model
- ✗ Belief projection: Random colors, no separation
- ✗ Value landscape: Noise, isolated peaks
- ✗ Belief evolution: Jumpy, chaotic paths
- ✗ Attention: Uniform weights (not learning)
- ✗ Metrics: No improvement, NaN/Inf

### Problem Diagnosis

| Symptom | Likely Cause | Fix |
|---------|------------|-----|
| Value landscape is noise | Value head not training | Increase `loss_weights['value']` |
| Belief projection is random | No belief structure learned | Check attention heatmaps |
| Uniform attention | Model not learning | Increase training iterations |
| Jumpy belief evolution | Unstable training | Lower learning rate |
| Isolated clusters | Overfitting | Add regularization, dropout |

## Advanced Usage

### Custom Projections

Use t-SNE instead of PCA (slower but often better):

```python
visualizer = BeliefStateVisualizer(agent, config)
visualizer.plot_belief_projection(beliefs, outcomes, method='tsne')
```

### Belief Space Analysis

Extract statistics about belief space geometry:

```python
import numpy as np

# Collect beliefs from games
beliefs = []  # (n_samples, latent_dim)

# Analyze
variance = beliefs.var(axis=0)  # Per-dimension variance
mean = beliefs.mean(axis=0)     # Per-dimension mean

print(f"Belief variance: {variance.mean():.4f}")
print(f"Most active dims: {np.argsort(variance)[-5:]}")
```

### Attention Analysis

Extract attention patterns:

```python
from src.evaluation import BeliefStateVisualizer

# During evaluation
outputs = agent([obs])
attention = outputs['attention_weights']  # List of (B, H, L, L)

# Per layer
for layer_idx, attn_w in enumerate(attention):
    final_attn = attn_w[0, :, -1, :].mean(dim=0)  # Final pos attention
    entropy = -(final_attn * (final_attn + 1e-10).log()).sum()
    print(f"Layer {layer_idx} attention entropy: {entropy:.3f}")
```

## Dependencies

**Required for basic training**:
- torch
- numpy

**Required for visualizations**:
- matplotlib (for plots)
- scikit-learn (for PCA/t-SNE)

Install optional dependencies:
```bash
pip install matplotlib scikit-learn
```

If not installed, visualizations will be skipped with a warning.

## Output Structure

```
logs/experiment_name/
├── training.log
├── metrics.json
├── checkpoint_iter*.pt
├── evaluation_results.json
└── visualizations/
    ├── training_summary.png      ← Overall metrics
    ├── belief_projection_pca.png  ← Belief space
    ├── value_landscape.png        ← Value function
    ├── belief_evolution.png       ← Trajectories
    ├── attention_heatmap_L*.png   ← Attention patterns
    └── belief_report.json         ← Visualization metadata
```

## Tips & Tricks

### 1. Monitor Training in Real-time
```bash
# Terminal 1: Train
python main.py --num-iterations 100

# Terminal 2: Watch metrics
watch -n 5 'tail -5 logs/*/metrics.json'
```

### 2. Compare Experiments
```python
import json
from pathlib import Path

experiments = ['exp1', 'exp2', 'exp3']
for exp in experiments:
    with open(f'logs/{exp}/metrics.json') as f:
        metrics = json.load(f)
    print(f"{exp}: reward={metrics['game_reward'][-1]:.3f}")
```

### 3. Generate Video
Create an animation of belief evolution:

```python
import matplotlib.animation as animation

# Collect beliefs from one game
beliefs_2d = []  # projected beliefs

fig, ax = plt.subplots()
def animate(frame):
    ax.clear()
    ax.scatter(beliefs_2d[frame, 0], beliefs_2d[frame, 1], s=100, c='blue')
    ax.set_xlim(beliefs_2d[:, 0].min(), beliefs_2d[:, 0].max())
    ax.set_ylim(beliefs_2d[:, 1].min(), beliefs_2d[:, 1].max())

anim = animation.FuncAnimation(fig, animate, frames=len(beliefs_2d))
anim.save('belief_evolution.gif')
```

### 4. Export for Papers
```python
# Save high-quality images for publication
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')
```

## Gallery

### Example Outputs

**Good Training** (100 iterations):
```
training_summary.png:
  Game Reward: 0 → +10 ↗
  Policy Loss: 2.5 → 1.2 ↘
  Value Loss: 0.5 → 0.1 ↘

belief_projection_pca.png:
  Green cluster (upper) = winning states
  Red cluster (lower) = losing states
  Clear separation = learning worked

value_landscape.png:
  Smooth blue-to-red gradient
  Correlates with outcome
  No noise

belief_evolution.png:
  5 trajectories shown
  Smooth paths through belief space
  Different endings for different outcomes
```

## Troubleshooting

### "No module named matplotlib"
```bash
pip install matplotlib
```

### Visualizations not generated
Check logs:
```bash
tail logs/*/training.log | grep -i visual
```

### Attention heatmaps are solid (all same color)
Model not learning attention patterns. Increase training time or model capacity.

### t-SNE crashes with "No module named sklearn"
```bash
pip install scikit-learn
# Or fall back to PCA
visualizer.plot_belief_projection(beliefs, method='pca')
```

## Next Steps

1. **Generate visualizations**: `python main.py --eval`
2. **Check metrics**: `logs/<exp>/training_summary.png`
3. **Analyze beliefs**: `logs/<exp>/belief_projection_pca.png`
4. **Debug attention**: `logs/<exp>/attention_heatmap_*.png`
5. **Publish**: Use high-res exports for papers

---

**Visualizations make research more interpretable and debugging easier!**

For questions, check [ARCHITECTURE.md](../ARCHITECTURE.md) "Belief State Geometry" section.
