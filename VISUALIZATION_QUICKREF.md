# Pygame Visualization - Quick Reference

## One-Line Commands

```bash
# Test setup
python test_visualization.py

# Watch agent play (auto)
python visualize.py --auto

# Play manually (click buttons)
python visualize.py

# Load trained model
python visualize.py --model logs/poker_transformer_default/checkpoint.pth --auto

# Demo with screenshots
python demo_visualization.py --hands 5 --delay 2000
```

## Controls

| Key | Action |
|-----|--------|
| R | Reset game |
| A | Toggle auto-play |
| Space | Single step (auto mode) |
| Q | Quit |
| Mouse | Click action buttons (manual mode) |

## Panel Locations

```
Top Row:        [Game Info]  [Cards]       [Controls]
Middle Row:     [Action Hist] [Pot Info] [Belief] [Value] [Transition]
Bottom Row:     [Attention Heatmap]
Bottom Edge:    [FOLD] [CALL] [RAISE] [CHECK] buttons
```

## What Colors Mean

### Belief State
- **Red bar**: Positive activation
- **Blue bar**: Negative activation
- **Height**: Magnitude

### Value Function
- **Green**: Positive (expecting to win)
- **Red**: Negative (expecting to lose)

### Transition Model
- **Green**: Low prediction error (accurate)
- **Red**: High prediction error (surprise)

### Attention Heatmap
- **Blue**: Low attention
- **Red**: High attention

### Action History
- **Red**: FOLD
- **Green**: CALL
- **Yellow**: RAISE
- **Blue**: CHECK

## Key Metrics to Watch

1. **Value Function**: Should correlate with your card strength
2. **Belief State**: Updates after each opponent action
3. **Attention**: Should focus on opponent raises
4. **Transition Error**: Lower is better (model learning dynamics)

## Common Issues

| Issue | Solution |
|-------|----------|
| Window too small | `--width 1920 --height 1080` |
| Auto-play too fast | `--play-delay 3000` |
| Model not found | Train first: `python main.py --num-iterations 10` |
| Game stuck | Press `R` to reset |

## Files Reference

| File | Purpose |
|------|---------|
| `src/visualization/pygame_visualizer.py` | Main visualization code |
| `visualize.py` | Command-line interface |
| `test_visualization.py` | Test suite |
| `demo_visualization.py` | Quick demo with screenshots |
| `VISUALIZATION_GUIDE.md` | Comprehensive guide |
| `VISUALIZATION_IMPLEMENTATION.md` | Implementation details |

## Quick Tips

1. **First run**: Use `python test_visualization.py` to verify setup
2. **Presentations**: Use `demo_visualization.py` with screenshots
3. **Analysis**: Use slow auto-play `--play-delay 3000` to observe
4. **Training**: Run `python main.py` to get a trained model first
5. **Manual play**: Good for understanding the game rules

## Minimum Working Example

```bash
# 1. Train a quick model
python main.py --num-iterations 5

# 2. Visualize it
python visualize.py --model logs/poker_transformer_default/checkpoint.pth --auto --play-delay 2000
```

That's it! Full documentation in `VISUALIZATION_GUIDE.md`
