# ✨ New Features: Gameplay & Geometry Visualization

## Summary

After training completes, your system now automatically generates:

### 1. **Sample Game Visualizations** 🎮
- 2 complete games with visual analysis
- Shows value progression, policy entropy, belief magnitude, action sequence
- JSON records for post-analysis

### 2. **Belief State Geometry** 📊
- PCA & t-SNE projections of belief space
- 50 games analyzed to understand learned representations
- Outcome-colored to show model structure

### 3. **Everything Runs Automatically**
```bash
python main.py --device cpu --num-iterations 10 --eval
```

Done! Check `logs/poker_transformer_default/`:
- `training_summary.png` - training curves ✓ (existing)
- `belief_geometry.png` - geometry analysis ✨ NEW
- `games/` folder with visualizations ✨ NEW

---

## Quick Start

### Run with Visualizations

```bash
cd poker_bot
python main.py --device cpu --latent-dim 64 --num-heads 4 --num-layers 3 --batch-size 32 --num-iterations 10
```

### View Results

```bash
# Open the logs directory
open logs/poker_transformer_default/

# You'll see:
# - belief_geometry.png (2D projections of beliefs)
# - games/sample_game_0_visualization.png (game 1)
# - games/sample_game_1_visualization.png (game 2)
# - games/sample_game_*.json (raw game data)
```

---

## Output Files

```
logs/poker_transformer_default/
├── training_summary.png                    ← Training curves
├── belief_geometry.png                     ← Belief space analysis (NEW)
├── games/                                  ← Individual games (NEW)
│   ├── sample_game_0_visualization.png
│   ├── sample_game_0_record.json
│   ├── sample_game_1_visualization.png
│   └── sample_game_1_record.json
├── visualizations/
│   ├── attention_heatmaps/
│   ├── belief_projections/
│   └── value_landscapes/
├── checkpoints/
├── metrics.json
└── training.log
```

---

## What Each Visualization Shows

### Sample Game Visualization

**4 subplots showing 1 complete game:**

1. **Value Progression** (top)
   - Blue line with dots
   - Shows estimated game value at each step
   - Should roughly trend toward final outcome

2. **Policy Entropy** (middle left)
   - Green line
   - High entropy = uncertain what to do
   - Low entropy = confident in decision

3. **Belief Magnitude** (middle right)
   - Red line
   - L2 norm of belief vector
   - Tracks how "far" from origin

4. **Action Sequence** (bottom)
   - Colored bars for CHECK/FOLD/CALL/RAISE
   - Shows the actual moves made

### Belief Geometry

**2 projections side-by-side:**

- **Left (PCA)**: Linear dimensionality reduction
  - Shows principal components
  - % variance explained per axis
  
- **Right (t-SNE)**: Nonlinear dimensionality reduction
  - Better at showing clusters
  - May take longer to compute

**Both**: Colored by outcome
- 🟢 Green = P0 won
- 🔴 Red = P1 won

**Interpretation**:
- ✅ **Good**: Clear win/loss clusters
- ✅ **Good**: Smooth, structured point distribution
- ❌ **Bad**: All points random/scattered
- ❌ **Bad**: Only one color (not learning outcome differences)

---

## Files Created

**New Module**: `src/evaluation/gameplay.py`
- `GameRecorder` class - plays games and records beliefs
- `play_and_visualize_sample_game()` - runs games with visualizations
- `visualize_geometry()` - analyzes belief space structure

**Updated Files**:
- `main.py` - now calls visualization after training
- `src/evaluation/__init__.py` - exports new functions

**Documentation**: 
- `GAMEPLAY_GUIDE.md` - detailed interpretation guide

---

## Customization

### Change Number of Games

Edit `main.py` line with:
```python
play_and_visualize_sample_game(trainer.agent, config, log_dir, num_games=5)  # Play 5 games instead of 2
```

### Change Geometry Sample Size

Edit `main.py` line:
```python
visualize_geometry(trainer.agent, config, log_dir)  # Default: 50 games
```

Then edit `gameplay.py` line 178:
```python
for i in range(100):  # Use 100 games instead of 50
```

### Disable Visualizations

Comment out these lines in `main.py`:
```python
# play_and_visualize_sample_game(trainer.agent, config, log_dir, num_games=2)
# visualize_geometry(trainer.agent, config, log_dir)
```

---

## Manual Usage

```python
import torch
from pathlib import Path
from src.config.config import ExperimentConfig
from src.evaluation.gameplay import GameRecorder, visualize_geometry
from src.model import PokerTransformerAgent

# Load trained agent
config = ExperimentConfig(device='cpu')
agent = PokerTransformerAgent(config)
checkpoint = torch.load('logs/checkpoint_iter10.pt')
agent.load_state_dict(checkpoint['agent_state'])

# Play games
recorder = GameRecorder(agent, config, Path('logs/manual'))

# Play 3 games
for i in range(3):
    game = recorder.play_game(seed=100 + i)
    recorder.visualize_game(game, f'game_{i}')
    recorder.save_game_record(game, f'game_{i}')

# Analyze geometry
visualize_geometry(agent, config, Path('logs/manual'))
```

---

## What Did We Add?

| Component | Purpose | File |
|-----------|---------|------|
| GameRecorder | Play games & record beliefs | `gameplay.py` |
| play_and_visualize_sample_game | Batch game playing | `gameplay.py` |
| visualize_geometry | Belief space analysis | `gameplay.py` |
| Game visualization | 4-panel game analysis | `gameplay.py` |
| Geometry visualization | PCA & t-SNE plots | `gameplay.py` |

---

## Performance Notes

| Operation | Time | CPU/GPU |
|-----------|------|---------|
| Playing 1 game | ~0.2s | CPU |
| Visualizing 1 game | ~0.5s | CPU |
| Analyzing 50 games | ~1-2s | CPU |
| t-SNE (50 games) | ~0.5-1s | CPU |
| **Total for all** | ~3-5s | CPU |

✅ All runs on CPU efficiently!

---

## Troubleshooting

**Q: Visualizations not created?**
A: Check that matplotlib/sklearn are installed:
```bash
pip install matplotlib scikit-learn
```

**Q: Game visualization looks blank?**
A: Make sure model is initialized (check if losses decrease during training)

**Q: Geometry shows random scatter?**
A: Model is not learning features. Try:
- More iterations
- Higher learning rate
- Check if training loss is decreasing

**Q: Takes too long?**
A: Reduce geometry samples in `gameplay.py` line 178:
```python
for i in range(10):  # Instead of 50
```

---

## Next Steps

1. ✅ Run training: `python main.py`
2. ✅ Check visualizations: `open logs/poker_transformer_default/`
3. ✅ Interpret results: Read `GAMEPLAY_GUIDE.md`
4. ✅ Experiment: Modify hyperparameters and compare visualizations
5. ✅ Analyze: Extract belief vectors for deeper analysis

---

## Summary

You now have:
- 🎮 **Live game playback** with visualizations
- 📊 **Belief space geometry** analysis
- 🎨 **Publication-quality figures** automatically generated
- 📈 **Training + Gameplay + Geometry** all in one pipeline

**All automatic, all on CPU, all beautiful!** ✨
