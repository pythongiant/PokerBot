# 🎮 Sample Gameplay & Geometry Visualization Guide

## What's New

After training completes, the system now automatically:

1. ✅ **Plays 2 sample games** with belief tracking
2. ✅ **Visualizes each game** with:
   - Value function progression
   - Policy entropy (uncertainty) 
   - Belief state magnitude
   - Action sequence
3. ✅ **Analyzes belief geometry** from 50 games using:
   - PCA projection
   - t-SNE projection (optional)
   - Outcome coloring (win/loss)

---

## Output Structure

```
logs/poker_transformer_default/
├── checkpoints/
├── training_summary.png           (existing)
├── belief_geometry.png             ✨ NEW
└── games/                          ✨ NEW
    ├── sample_game_0_visualization.png
    ├── sample_game_0_record.json
    ├── sample_game_1_visualization.png
    └── sample_game_1_record.json
```

---

## Example Outputs

### 1. Sample Game Visualization

Shows a single game with 4 subplots:

**Top**: Value Function Progression
- Line plot showing estimated values over game steps
- Should follow learned strategy

**Middle Left**: Policy Entropy
- Measures uncertainty in action selection
- High entropy = uncertain, Low entropy = confident

**Middle Right**: Belief State Magnitude
- L2 norm of belief vector
- Shows how "far" from origin the model is representing

**Bottom**: Action Sequence
- Bar chart of actions taken
- Color-coded: CHECK (blue), FOLD (red), CALL (green), RAISE (orange)

### 2. Belief State Geometry

Two side-by-side projections:

**Left Panel**: PCA (Principal Component Analysis)
- Linear dimensionality reduction
- Shows variance explained by each component
- Points colored by outcome (green = win, red = loss)

**Right Panel**: t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Nonlinear dimensionality reduction
- Better at showing clusters
- Same outcome coloring

**Interpretation**:
- ✅ **Good**: Wins and losses are well-separated
- ✅ **Good**: No completely random noise
- ❌ **Bad**: All points same color (model not learning)
- ❌ **Bad**: Dense cloud with no structure

---

## Running with New Features

### Automatic (Recommended)

Training now runs game visualization automatically:

```bash
python main.py --device cpu --latent-dim 64 --num-heads 4 --num-layers 3 --batch-size 32 --num-iterations 10
```

After training:
1. Training summary visualization ✓ (existing)
2. Sample game 0 + visualization ✓ (new)
3. Sample game 1 + visualization ✓ (new)
4. Belief geometry analysis ✓ (new)

### Manual

Play and visualize games on demand:

```python
from pathlib import Path
from src.evaluation.gameplay import play_and_visualize_sample_game, visualize_geometry
from src.model import PokerTransformerAgent
from src.config import ExperimentConfig

# Load trained agent
config = ExperimentConfig(device='cpu')
agent = PokerTransformerAgent(config)
agent.load_state_dict(torch.load('logs/checkpoint.pt'))

# Play 5 games
play_and_visualize_sample_game(agent, config, Path('logs/manual_games'), num_games=5)

# Analyze geometry
visualize_geometry(agent, config, Path('logs/manual_games'))
```

---

## Game Record Format

Each game is saved as JSON with:

```json
{
  "seed": 42,
  "steps": 4,
  "actions": ["FOLD", "CHECK", ...],
  "observations": [...],
  "beliefs": [[...belief_vector...], ...],
  "values": [0.5, 0.3, ...],
  "policies": [[...policy_logits...], ...],
  "legal_actions": [["CHECK", "RAISE"], ...],
  "game_state": {
    "private_cards": [0, 2],
    "public_cards": [],
    "stacks": [100, 95],
    "pot": 5,
    "is_terminal": true,
    "payoffs": [-2.5, 2.5]
  },
  "rewards": [-2.5, 2.5]
}
```

---

## Interpreting the Visualizations

### Value Function Progression

| Pattern | Meaning |
|---------|---------|
| Smooth trend ✓ | Learning is working |
| Wildly oscillating ✗ | Training instability |
| Flat zero ✗ | Model not initialized properly |
| Consistent sign | Good - predicts win/loss |

### Policy Entropy

| Pattern | Meaning |
|---------|---------|
| Decreasing ✓ | Model becoming more confident |
| High > 1.0 | Too much exploration/randomness |
| Sudden drops | Discovery of winning strategy |

### Belief State Magnitude

| Pattern | Meaning |
|---------|---------|
| Smooth trajectory ✓ | Stable learning dynamics |
| Spikes ✗ | Potential gradient issues |
| Trending up/down | Model drift |

### Belief Geometry (PCA/t-SNE)

| Pattern | Meaning |
|---------|---------|
| Clear win/loss clusters ✓✓ | Excellent feature learning |
| Partial separation ✓ | Good progress |
| Random scatter ✗ | Model not learning features |
| Single point | All games identical (check if deterministic) |

---

## Troubleshooting

### Issue: "matplotlib not installed"

**Fix**: Install optional visualization dependencies
```bash
pip install matplotlib scikit-learn
```

**If you skip this**: Visualizations are disabled but training continues

### Issue: Geometry looks random

**Possible causes**:
- Model is untrained (check loss values)
- Model is undertrained (run more iterations)
- Training hyperparameters need tuning

**Solution**: Run more iterations or increase learning rate

### Issue: All games look identical

**Possible cause**: Model is deterministic/not learning

**Check**: Look at value outputs - should vary by game

---

## Advanced Usage

### Extract belief vectors for external analysis

```python
from src.evaluation.gameplay import GameRecorder

recorder = GameRecorder(agent, config, Path('logs'))
game = recorder.play_game(seed=123)

beliefs = np.array(game['beliefs'])  # (num_steps, latent_dim)
values = np.array(game['values'])
policies = np.array(game['policies'])

# Now use for external ML/analysis
```

### Custom visualization

```python
import matplotlib.pyplot as plt
from src.evaluation.gameplay import GameRecorder

recorder = GameRecorder(agent, config, Path('logs'))
game = recorder.play_game(seed=123)

beliefs = np.array(game['beliefs'])

# Plot belief trajectory
plt.figure()
for i in range(min(5, beliefs.shape[1])):  # First 5 dimensions
    plt.plot(beliefs[:, i], label=f'dim_{i}')
plt.legend()
plt.show()
```

---

## Summary

The new features give you:

| Feature | Purpose | Output |
|---------|---------|--------|
| Sample Games | See model behavior in action | PNG + JSON per game |
| Value Progression | Track value estimates | Graph showing trends |
| Policy Entropy | Measure confidence | Graph showing uncertainty |
| Belief Magnitude | Track learning dynamics | Graph showing trajectory |
| PCA Geometry | Linear view of beliefs | 2D scatter plot |
| t-SNE Geometry | Nonlinear clusters | 2D scatter plot |

**Total new files per training**: 6-8 visualizations + 2 JSON records

✨ **Use these to understand what your model learned!**
