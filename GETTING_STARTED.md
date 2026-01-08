# Getting Started with Poker Transformer

Welcome! This guide will get you up and running in 5 minutes.

## Installation

### 1. Prerequisites
- Python 3.8+
- pip or conda

### 2. Install Dependencies
```bash
cd poker_bot
pip install -r requirements.txt
```

For GPU (optional):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Validation (1 min)

Verify everything works before training:

```bash
python quickstart.py
```

This will:
- ✓ Run 2 quick training iterations
- ✓ Evaluate agent vs random baseline
- ✓ Save logs to `logs/quickstart/`
- ✓ Exit with success message

Expected output:
```
SUCCESS! ✓

Next steps:
  1. Check logs in: ./logs/quickstart/
  2. Run full training: python main.py --num-iterations 100
  3. Explore examples: python examples/examples.py --example 1
```

## Running Training

### Default Setup (Recommended)

```bash
python main.py --num-iterations 100 --eval
```

This trains for 100 iterations and evaluates at the end.

**Runtime**: ~30 minutes on CPU, ~5 minutes on GPU

**Output**:
- Logs in `logs/poker_transformer_default/`
- Model checkpoints every 10 iterations
- Metrics and evaluation results

### Custom Configuration

```bash
# Larger, more capable model
python main.py \
  --name "my_experiment" \
  --latent-dim 128 \
  --num-layers 6 \
  --num-iterations 200 \
  --learning-rate 1e-4 \
  --eval

# Fast test run
python main.py \
  --name "quick_test" \
  --num-iterations 10 \
  --games-per-iteration 32 \
  --batch-size 16

# GPU training
python main.py \
  --device cuda \
  --num-iterations 500 \
  --batch-size 64 \
  --learning-rate 1e-3
```

## Exploring Results

After training, check the logs:

```bash
# View training progress
cat logs/<experiment_name>/training.log

# Check metrics (JSON)
cat logs/<experiment_name>/metrics.json

# Evaluation results
cat logs/<experiment_name>/evaluation_results.json
```

### Understanding Metrics

In `metrics.json`:
- `game_reward`: Average payoff per game (should increase)
- `policy_loss`: KL divergence from targets (should decrease)
- `value_loss`: MSE from targets (should decrease)
- `iteration`: Training iteration number

**Healthy training**:
- Game reward trending up
- Losses trending down
- No NaN or inf values
- Stable gradient norms

## Examples

Learn by doing! Explore different training scenarios:

```bash
# Example 1: Basic training
python examples/examples.py --example 1

# Example 2: Larger model
python examples/examples.py --example 2

# Example 3: Ablation study
python examples/examples.py --example 3

# Example 4: Belief state analysis
python examples/examples.py --example 4

# Example 5: Compare search methods
python examples/examples.py --example 5
```

Each example is self-contained and demonstrates a specific workflow.

## Understanding the Model

Read these in order:

1. **README.md** (5 min): High-level overview
2. **ARCHITECTURE.md** (20 min): Deep dive into components
3. **ROADMAP.md** (10 min): Future extensions and research

## Debugging

### "Model not training"
```bash
# Check if loss is decreasing
tail -20 logs/*/metrics.json

# Run validation
python validate.py

# Check configuration
grep -A 10 "loss_weights" logs/*/training.log
```

### "Low win rate against random"
- Try longer training: `--num-iterations 500`
- Increase model capacity: `--latent-dim 128 --num-layers 6`
- Check learning rate: `--learning-rate 1e-3` (not too high/low)

### "Out of memory"
- Reduce batch size: `--batch-size 16`
- Reduce games per iteration: `--games-per-iteration 64`
- Reduce latent dim: `--latent-dim 32`

### "CUDA out of memory"
```bash
# Fall back to CPU
python main.py --device cpu --num-iterations 50
```

## Next Steps

### Level 1: Reproduce Results
- [x] Run `quickstart.py`
- [x] Run `main.py` with defaults
- [x] Check metrics in `logs/`

### Level 2: Understand the Code
- [ ] Read ARCHITECTURE.md
- [ ] Trace through `main.py` → `trainer.py` → `agent.py`
- [ ] Study `src/environment/kuhn.py`

### Level 3: Run Experiments
- [ ] Try different hyperparameters
- [ ] Run ablations (disable value head, transition, etc.)
- [ ] Compare vs baselines

### Level 4: Extend the System
- [ ] Add opponent modeling
- [ ] Implement full MCTS
- [ ] Extend to Leduc poker
- [ ] Publish results!

## Common Questions

### Q: How long does training take?
- **CPU**: ~1 min/iteration (100 iterations = ~1.5 hours)
- **GPU**: ~10 sec/iteration (100 iterations = ~15 min)

### Q: What's a good win rate against random?
- Random baseline: 50% (expected for fair game)
- Good agent: 60-70% (beats random strategy)
- Optimal (Nash): ~100% if exploitable

### Q: Can I run on my laptop?
- **Yes!** CPU training works fine
- Slower but stable (PyTorch handles this)
- Recommended: use `--num-iterations 50` for testing

### Q: How do I use a trained model?
```python
import torch
from src.model import PokerTransformerAgent
from src.config import ExperimentConfig

# Load model
checkpoint = torch.load('logs/best_model.pt')
config = ExperimentConfig()
agent = PokerTransformerAgent(config)
agent.load_state_dict(checkpoint['agent_state'])

# Play a game
from src.environment import KuhnPoker, Action

env = KuhnPoker()
game_state, obs = env.reset()

while not game_state.is_terminal:
    with torch.no_grad():
        belief, _ = agent.encode_belief([obs])
        policy_logits = agent.predict_policy(belief)
    
    legal_actions = env.get_legal_actions(game_state.current_player)
    probs = torch.softmax(policy_logits[0], dim=-1).numpy()
    action = legal_actions[np.argmax(probs[legal_actions])]
    
    game_state, obs, _ = env.step(game_state.current_player, Action(action))
```

### Q: How do I contribute improvements?
1. Fork/clone the repo
2. Create a feature branch
3. Make changes in modular way
4. Add tests/validation
5. Update documentation
6. Submit PR with clear description

## Architecture at a Glance

```
Game History (cards + actions)
    ↓
[Causal Transformer] → Belief State z_t
    ↓
    ├─→ [Value Head] → Reward estimate
    ├─→ [Policy Head] → Action selection
    ├─→ [Transition] → Next belief z_{t+1}
    └─→ [Opponent Range] → Hand estimation
    
    ↓
[Self-Play] → Trajectories + Payoffs
    ↓
[Search/Rollout] → Improved targets
    ↓
[Training] → Update model parameters
    ↓
[Evaluation] → Check performance
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point |
| `quickstart.py` | Validation script |
| `validate.py` | Component tests |
| `src/config/config.py` | Configuration |
| `src/environment/kuhn.py` | Poker game |
| `src/model/agent.py` | Main model |
| `src/training/trainer.py` | Training loop |
| `src/evaluation/evaluator.py` | Evaluation |
| `examples/examples.py` | Usage examples |
| `README.md` | Overview |
| `ARCHITECTURE.md` | Technical deep dive |
| `ROADMAP.md` | Future work |

## Resources

- **Paper Reading**: MuZero (Schaal & Silver), CFR (Zinkevich et al.), Transformers (Vaswani et al.)
- **Game Theory**: Kuhn poker is solvable via CFR (can compare results)
- **RL Books**: Sutton & Barto for foundations
- **Poker Theory**: "The Mathematics of Poker" by Chen & Ankenman

## Support

Having issues?

1. Check this guide (FAQ section)
2. Review error message in `logs/*/training.log`
3. Run `python validate.py` to test components
4. Read relevant `.md` file (README/ARCHITECTURE/ROADMAP)
5. Inspect code comments (they explain the "why")

## What's Next?

Once you're comfortable:
- Implement MCTS fully
- Extend to Leduc poker
- Add opponent modeling
- Compute exploitability
- Write up results as research paper!

---

**Happy training! 🎉**

Questions? Check ARCHITECTURE.md for deeper explanations.
