# Poker Transformer: Learned Belief State + Value + Policy

A research-grade Poker AI built with a **Transformer-based belief encoder**, **learned transition dynamics**, and **value function**, inspired by MuZero + CFR but adapted for partial observability.

## 🎯 Objective

Build a Poker agent that:

- **Encodes** the entire observable game history (cards + betting) into a latent belief state using a causal Transformer
- **Learns** latent dynamics $z_{t+1} = g_\theta(z_t, a_t)$ to model opponent behavior
- **Predicts** value $V_\theta(z_t) \approx \mathbb{E}[\text{EV} \mid z_t]$ under uncertainty
- **Outputs** a policy $\pi_\theta(a \mid z_t)$ over actions
- **Trains** via self-play using search/rollout targets (MuZero-style)

The system is designed for research experiments, not as a toy demo.

## 🏗 Architecture

### Belief State Encoding
```
Observable History (cards + betting)
           ↓
    [CausalTransformer]
           ↓
   z_t: Latent Belief State
           ↓
    ├─→ [ValueHead] → V_θ(z_t)
    ├─→ [PolicyHead] → π_θ(a | z_t)
    ├─→ [TransitionModel] → z_{t+1} = g_θ(z_t, a_t)
    └─→ [OpponentRangePredictor] → P(opp_card | z_t) [optional]
```

### Key Components

1. **BeliefStateTransformer**: Causal (masked) multi-head attention encoder
   - Input embeddings: card + action + bet amount
   - Positional encoding for temporal structure
   - Output: latent belief state (configurable dimension)

2. **LatentTransitionModel**: Learns belief dynamics
   - Input: $z_t$ + one-hot action
   - Output: $z_{t+1}$
   - Implicitly learns opponent strategies and chance events

3. **ValueHead**: Counterfactual EV estimation
   - Input: belief state $z_t$
   - Output: scalar value
   - Trained against bootstrapped returns

4. **PolicyHead**: Action selection
   - Input: belief state $z_t$
   - Output: logits over [FOLD, CALL, RAISE, CHECK]
   - Masked to legal actions
   - Trained via cross-entropy against search targets

5. **OpponentRangePredictor** (optional): Interpretability
   - Predicts opponent's card distribution from belief
   - Useful for analyzing what the model learned

## 📊 Training

### Self-Play Loop
```
for iteration in 1..N:
    1. Run B games of self-play (with exploration)
    2. [Optional] Run MCTS/rollouts to generate targets
    3. Create training batch from game trajectories
    4. Compute losses:
       L = λ_π · KL(π* || π_θ)
         + λ_v · (V* - V_θ)²
         + λ_d · ||z_{t+1} - g_θ(z_t, a_t)||²
         + [λ_opp · opponent_range_loss]
    5. Optimize with gradient clipping & mixed precision
    6. Checkpoint & evaluate
```

### Targets
- **Policy target**: From MCTS visit counts or improved rollout policy
- **Value target**: Monte Carlo returns or bootstrapped value
- **Transition target**: Predicted next belief (requires encoding future states)

## 🚀 Quick Start

### Installation
```bash
# Clone/download the repo
cd poker_bot

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Default configuration
python main.py --num-iterations 100

# Custom configuration
python main.py \
  --name "exp_larger_model" \
  --latent-dim 128 \
  --num-layers 4 \
  --learning-rate 1e-3 \
  --num-iterations 500 \
  --games-per-iteration 256

# With evaluation
python main.py --num-iterations 100 --eval --eval-games 200
```

### Command-Line Arguments

**Experiment**:
- `--name`: Experiment identifier
- `--log-dir`: Logging directory
- `--device`: `cpu` or `cuda`
- `--seed`: Random seed (reproducibility)

**Model Architecture**:
- `--latent-dim`: Belief state dimension (default: 64)
- `--num-heads`: Attention heads (default: 4)
- `--num-layers`: Transformer layers (default: 3)
- `--ff-dim`: Feed-forward hidden dim (default: 256)
- `--transition-type`: `deterministic` or `probabilistic`

**Training**:
- `--num-iterations`: Total iterations (default: 100)
- `--games-per-iteration`: Games per iteration (default: 128)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--optimizer`: `adam` or `sgd`
- `--weight-decay`: L2 regularization (default: 1e-5)

**Search**:
- `--search-type`: `mcts` or `rollout`
- `--num-simulations`: MCTS simulations (default: 50)
- `--rollout-depth`: Rollout depth (default: 10)

**Evaluation**:
- `--eval`: Run full evaluation after training
- `--eval-games`: Number of eval games
- `--checkpoint`: Load checkpoint for eval

### Output

Training outputs are saved to `logs/<experiment_name>/`:
- `training.log`: Training logs
- `checkpoint_iter*.pt`: Model checkpoints
- `metrics.json`: Training metrics
- `evaluation_results.json`: Eval results (if `--eval`)
- `visualizations/`: Auto-generated plots (belief projections, value landscapes, attention heatmaps)

## 📈 Key Experiments

### 1. Basic Training
Verify the system trains stably in self-play.

```bash
python main.py \
  --name "basic_training" \
  --num-iterations 50 \
  --games-per-iteration 64 \
  --latent-dim 64
```

**Expected**: Positive average reward over iterations (learning).

### 2. Ablation: No Transition Model
Disable transition learning to measure its contribution.

```bash
# Modify config to skip transition loss
python main.py --name "ablation_no_transition"
```

### 3. Ablation: No Value Head
Train with policy only.

```bash
# Modify config: loss_weights['value'] = 0.0
python main.py --name "ablation_no_value"
```

### 4. Search-Based Targets vs. RL
Compare MCTS targets vs. raw self-play rollouts.

```bash
python main.py --search-type mcts --name "with_mcts"
python main.py --search-type rollout --name "rollout_only"
```

### 5. Belief State Analysis
Extract and visualize belief geometry.

```bash
python main.py --eval --name "belief_analysis"
# Examine attention patterns and value landscape in evaluation_results.json
```

### 6. Exploitability
Compare against game-theoretic baseline (requires CFR solver).

```bash
# TODO: Integrate nashpy or gambit for exact exploitability
```

## 🧠 Model Details

### Belief State Visualization

Training automatically generates rich visualizations:

- **Training Metrics Dashboard**: Game reward, policy loss, value loss over iterations
- **Belief State Projections**: 2D visualization of latent belief space (colored by outcome)
- **Value Function Landscape**: How value estimates vary across belief space
- **Belief Evolution**: How beliefs change during individual games
- **Attention Heatmaps**: Which history positions the model attends to

All visualizations are saved to `logs/<experiment>/visualizations/`.

See [VISUALIZATIONS.md](VISUALIZATIONS.md) for interpretation guide.

The latent belief state $z_t$ encodes:
- **Own card strength** (embedded in initial token)
- **Opponent's likely hand range** (from betting patterns + attention)
- **Game progress** (street, pot size)
- **Uncertainty** (implicit in distribution over latent dimensions)

**Analysis utilities** (in `src/model/agent.py: BeliefStateGeometry`):
- `attention_to_opponent_actions()`: Extract opponent-focused attention
- `belief_state_variance()`: Information spread across dimensions
- `value_landscape()`: Value function geometry
- `attention_flow_analysis()`: How information flows through layers

### Causal Attention

Why causal (masked) attention?
- **Temporal ordering**: Actions at $t$ don't see future actions
- **Belief updates**: Opponent strategies inferred from past, not future
- **Partial observability**: Only own card + public history visible

### Transition Dynamics

The transition model $g_\theta(z_t, a_t) \to z_{t+1}$ learns:
- How beliefs update given new information
- Implicit opponent mixed strategies
- Reactions to different board runouts

For now: **deterministic transitions**. Can extend to probabilistic (mean + variance).

## 📁 Project Structure

```
poker_bot/
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
├── README.md                        # This file
│
└── src/
    ├── __init__.py
    ├── config/
    │   ├── config.py                # Configuration dataclasses
    │   └── __init__.py
    │
    ├── environment/
    │   ├── kuhn.py                  # Kuhn poker implementation
    │   └── __init__.py
    │
    ├── model/
    │   ├── transformer.py           # Causal Transformer encoder
    │   ├── heads.py                 # Value, Policy, Transition heads
    │   ├── agent.py                 # Unified PokerTransformerAgent
    │   └── __init__.py
    │
    ├── training/
    │   ├── search.py                # MCTS, rollouts, self-play
    │   ├── trainer.py               # Main training loop
    │   └── __init__.py
    │
    └── evaluation/
        ├── evaluator.py             # Head-to-head, exploitability, analysis
        └── __init__.py
```

## 🔬 Research Extensions

### 1. Kuhn → Leduc Poker
Extend from 3-card to 6-card with 2 streets.

**Changes**:
- Update `environment/` to handle flop + turn
- Increase latent dim and model capacity
- Longer sequences → more Transformer layers

### 2. Hierarchical RL
Add option framework: high-level (fold/play) vs. low-level (bet sizing).

### 3. CFR Baseline Comparison
Solve Kuhn poker exactly with CFR and compare strategies.

### 4. Belief Probing
Use attention analysis to extract estimated opponent ranges.

```python
from src.model import BeliefStateGeometry
geometry = BeliefStateGeometry(agent)
opponent_attn = geometry.get_attention_to_opponent_actions(...)
```

### 5. Multi-Player Variants
Extend to 3+ players (exponentially harder).

### 6. Imitation Learning
Pre-train on expert games (CFR baseline) before self-play.

## 📚 Theory Grounding

### Partially Observable Markov Game (POMG)
- **State**: Full game state (including opponent cards)
- **Observation**: Public state + own cards (no opponent cards)
- **Belief**: Distribution over opponent hands given history

### Belief-State MDP
The Transformer learns to approximate the belief-state MDP where:
- **States**: Belief states $z_t = f_\theta(o_{1:t})$
- **Dynamics**: $z_{t+1} = g_\theta(z_t, a_t)$ learned
- **Value**: $V_\theta(z_t)$ learned

### Loss Decomposition
```
L_total = λ_π · L_policy + λ_v · L_value + λ_d · L_dynamics + λ_opp · L_opponent
```

- **L_policy**: KL divergence from search targets
- **L_value**: MSE against bootstrapped returns
- **L_dynamics**: Transition model accuracy (if supervised signal available)
- **L_opponent**: Range prediction (optional regularization)

## ⚠️ Known Limitations

1. **Kuhn Poker Only**: Extends to Leduc but not larger games
2. **No Bet Sizing**: Fixed bet amounts (extension: continuous action space)
3. **Deterministic Transitions**: Could add stochasticity for full belief distribution
4. **No Importance Sampling**: Off-policy training not yet implemented
5. **Limited Search**: MCTS is simplified; could integrate alpha-zero-style search

## 🐛 Debugging Tips

**Model not training**:
- Check loss is decreasing in `metrics.json`
- Verify belief states are changing with `belief_state_variance()`
- Inspect attention patterns in `evaluation_results.json`

**Poor performance vs. random**:
- Increase model capacity: `--latent-dim 128 --num-layers 4`
- Increase training iterations: `--num-iterations 500`
- Check if policy is being updated (policy loss → 0?)

**Reproducibility**:
- Set `--seed` and use deterministic operations
- Log config to `training.log`

## 📖 References

- **MuZero**: Schaal & Silver et al. - Mastering games via latent dynamics models
- **CFR**: Zinkevich et al. - Regret minimization in games with incomplete information
- **Transformers**: Vaswani et al. - Attention is all you need
- **Belief-State Games**: Jones et al. - Effective policies for POMDPs

## ✅ Checklist for Research Use

- [x] Clean, modular code
- [x] Clear separation: environment, model, training, evaluation
- [x] Minimal poker environment (Kuhn)
- [x] Causal Transformer belief encoder
- [x] Learned transition model
- [x] Value and policy heads
- [x] Self-play training loop
- [x] Multiple evaluation metrics
- [x] Belief state geometry analysis
- [x] Extensible for research
- [x] Comments explaining why, not just what
- [x] No hardcoded hacks

## 🎓 Citation

If you use this code in research, cite:

```
@misc{poker_transformer_2024,
  title={Poker Transformer: Belief State + Value + Policy},
  author={Research Engineer},
  year={2024},
  howpublished={\url{https://github.com/...}}
}
```

## 📞 Support

For issues or questions:
1. Check logs in `logs/<experiment>/training.log`
2. Verify config in `logs/<experiment>/metrics.json`
3. Review code comments in respective modules
4. Run diagnostic: `python main.py --eval --name debug`

---

**Status**: Alpha (research-grade, ready for experiments)

**Python**: 3.8+  
**PyTorch**: 2.0+
