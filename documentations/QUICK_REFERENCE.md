# Quick Reference & Visual Guide

## System Architecture (Visual)

```
┌─────────────────────────────────────────────────────────────────┐
│                      POKER TRANSFORMER AGENT                    │
└─────────────────────────────────────────────────────────────────┘

INPUT: Game Observation
┌────────────────────────────────────────────────┐
│ own_card: 0|1|2                                │
│ action_history: [(player, action, amount), ...] │
│ current_player: 0|1                            │
│ stacks: [100, 100]                             │
│ pot: amount                                    │
└────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────┐
│    BELIEF STATE TRANSFORMER                    │
│                                                │
│  Embedding Layer                               │
│    ├─ Card: one-hot → d_card dims              │
│    ├─ Action: one-hot → d_action dims          │
│    └─ Bet: linear projection → d_bet dims      │
│                                                │
│  Sequence Stack:                               │
│    └─ [own_card] + [action1, action2, ...]    │
│                                                │
│  Causal Transformer (N layers)                │
│    ├─ Multi-head attention (masked)           │
│    ├─ Feed-forward network                    │
│    ├─ Layer norm + residual                   │
│    └─ Extract final token                     │
│                                                │
│  Output: z_t (latent_dim,)                     │
└────────────────────────────────────────────────┘
        ↓
┌──────────┬──────────┬──────────┬──────────────┐
│          │          │          │              │
↓          ↓          ↓          ↓              ↓
┌────┐  ┌────┐  ┌────┐  ┌───────────┐  ┌──────┐
│Val │  │Pol │  │Tra │  │Opponent   │  │ ...  │
│Hd  │  │Hd  │  │Hd  │  │Range      │  │      │
├────┤  ├────┤  ├────┤  ├───────────┤  └──────┘
│    │  │    │  │    │  │ Optional  │
│ V  │  │ π  │  │ z' │  │ Head      │
│ᶿ  │  │ᶿ  │  │ᶿ  │  │           │
└────┘  └────┘  └────┘  └───────────┘
  ↓       ↓       ↓          ↓
  │       │       │          │
  │     Policy   │      Opponent
  │    Distribution  │      Hand
  │     (4-dim)     │   Distribution
  │               │   (3-dim, Kuhn)
  │               │
Value Estimate    Next Belief State
(scalar)          (latent_dim,)

OUTPUT: Actions + Targets
├─ Policy: softmax(π_logits) → action probabilities
├─ Value: V scalar → estimated payoff
├─ Next Belief: z' → for rollouts
└─ Opponent Range: P(card|z) → interpretability
```

## Training Loop (Pseudocode)

```python
for iteration in range(num_iterations):
    # PHASE 1: Collect self-play games
    games = []
    for _ in range(games_per_iteration):
        game_state, obs = env.reset()
        while not game_state.is_terminal:
            # Get current player's action
            belief = agent.encode_belief(obs)
            action ~ π_θ(·|belief)
            
            # Step environment
            game_state, obs = env.step(action)
        
        games.append({
            'observations': [...],
            'actions': [...],
            'payoff': game_state.payoffs
        })
    
    # PHASE 2: Train model
    for batch in shuffle(games):
        # Forward pass
        beliefs = agent.encode_belief(batch.observations)
        values = value_head(beliefs)           # (batch, 1)
        logits = policy_head(beliefs)          # (batch, 4)
        next_beliefs = transition(beliefs, a)  # (batch, latent_dim)
        
        # Targets (from payoffs or search)
        targets_v = batch.payoff              # (batch,)
        targets_π = improved_policy(batch)    # (batch, 4)
        
        # Compute losses
        loss_v = MSE(values, targets_v)       # Value loss
        loss_π = KL(π, targets_π)             # Policy loss
        loss_d = MSE(next_beliefs_actual, next_beliefs)  # Transition
        
        loss_total = λ_v * loss_v + λ_π * loss_π + λ_d * loss_d
        
        # Optimize
        loss_total.backward()
        clip_gradients()
        optimizer.step()
    
    # PHASE 3: Evaluate
    if iteration % eval_freq == 0:
        win_rate = evaluate_vs_random()
        belief_stats = analyze_belief_geometry()
        print(f"Iteration {iteration}: WR={win_rate:.1%}")
```

## Configuration Quick Reference

```yaml
# Model Size
latent_dim: 64            # Larger = more capacity
num_heads: 4              # More = more parallelism
num_layers: 3             # Deeper = more computation
ff_dim: 256               # Wider FFN

# Training Dynamics
num_iterations: 100       # More = better convergence
games_per_iteration: 128  # More = better targets
batch_size: 32            # Larger = faster but uses more memory
learning_rate: 1e-3       # Typical range: 1e-4 to 1e-2

# Loss Weights (tune these for ablations)
loss_weights:
  policy: 1.0             # Always on
  value: 1.0              # Always on
  transition: 0.5         # Optional: 0.0 to disable
  opponent_range: 0.1     # Optional: 0.0 to disable
```

## Hyperparameter Tuning Guide

### For Faster Training
```bash
python main.py \
  --latent-dim 32 \
  --num-layers 2 \
  --ff-dim 128 \
  --batch-size 64 \
  --games-per-iteration 256 \
  --learning-rate 1e-2
```

### For Better Convergence (Slower)
```bash
python main.py \
  --latent-dim 256 \
  --num-layers 8 \
  --ff-dim 512 \
  --batch-size 16 \
  --games-per-iteration 64 \
  --learning-rate 1e-4
```

### For Laptop/Limited Resources
```bash
python main.py \
  --device cpu \
  --latent-dim 32 \
  --num-layers 2 \
  --num-iterations 20 \
  --games-per-iteration 32 \
  --batch-size 8
```

## Data Structures at a Glance

### Observation (from Environment)
```python
obs = ObservableState(
    own_card: int,              # 0, 1, or 2 (Kuhn)
    public_cards: List[int],    # [] for Kuhn, [card] for Leduc
    action_history: List[(player: int, action: Action, amount: int)],
    current_player: int,        # 0 or 1
    stacks: List[int],          # [100, 100]
    pot: int,                   # Chips in pot
    street: int,                # 0 (Kuhn), 0-1 (Leduc)
)
```

### Belief State (latent)
```python
z_t: torch.Tensor           # Shape: (latent_dim,)
                            # Encodes all history compactly
                            # Not interpretable a priori
                            # Learned via training
```

### Model Outputs
```python
outputs = {
    'belief_states': (batch, latent_dim),       # Encoded beliefs
    'values': (batch, 1),                       # Value estimates
    'policy_logits': (batch, 4),                # Action logits
    'opponent_range_logits': (batch, 3),        # Hand distribution (optional)
    'attention_weights': List[(batch, heads, seq_len, seq_len)],
    'final_hidden': (batch, seq_len, latent_dim),
}
```

### Training Batch
```python
batch = {
    'observations': List[ObservableState],  # Variable length
    'policy_targets': (batch, 4),           # From search
    'value_targets': (batch,),              # From rollouts
}
```

## Common Commands

```bash
# Quick validation (1 min)
python quickstart.py

# Full component test (5 min)
python validate.py

# Basic training (15 min on GPU)
python main.py --num-iterations 50

# Research-grade training (1-2 hours on GPU)
python main.py --num-iterations 500 --latent-dim 128 --num-layers 6

# With evaluation
python main.py --num-iterations 100 --eval --eval-games 200

# Ablation: no transition model
python main.py --name "no_transition" \
  # [manually modify config to set loss_weights['transition']=0]

# Run examples
python examples/examples.py --example 1    # Basic
python examples/examples.py --example 4    # Belief analysis

# Check logs
tail -20 logs/*/training.log
cat logs/*/metrics.json | python -m json.tool
```

## Interpreting Results

### Good Training Metrics
```json
{
  "game_reward": [0.1, 0.15, 0.2, 0.25, ...],    # ↑ Going up
  "policy_loss": [2.5, 2.3, 2.1, 2.0, ...],      # ↓ Going down
  "value_loss": [0.5, 0.45, 0.4, 0.35, ...],     # ↓ Going down
  "iteration": [0, 1, 2, 3, ...]
}
```

### Poor Training Indicators
```json
{
  "game_reward": [0.1, 0.1, 0.1, 0.1, ...],      # Flat (no learning)
  "policy_loss": ["nan", "nan", ...],             # NaN (numerical issue)
  "value_loss": [1.0, 2.0, 3.0, ...],            # ↑ Getting worse
}
```

### Evaluation Quality
| Metric | Interpretation |
|--------|-----------------|
| Win rate = 50% | Random = agent performs as random |
| Win rate = 60-70% | Good = agent beats random |
| Win rate = 80%+ | Strong = agent exploits well |
| Reward = +5 chips | Good (Kuhn, +/-100 possible) |

## Attention Analysis (What Model Learns)

```python
# Extract attention weights
outputs = agent([obs])
attn_weights = outputs['attention_weights']  # List of (B, H, L, L)

# Per layer
for layer_idx, attn_w in enumerate(attn_weights):
    final_attn = attn_w[batch_idx, head_idx, -1, :]  # Where does last token attend?
    opponent_attn = final_attn[opponent_action_positions]  # Attention to opponent moves?
    print(f"Layer {layer_idx}: opponent_attention = {opponent_attn.mean():.3f}")

# Good sign: attention to opponent actions increases over layers
# Bad sign: all attention on one position (collapsing)
```

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| CUDA out of memory | Batch too large | Reduce `--batch-size` |
| NaN in loss | Numerical instability | Lower `--learning-rate` |
| Low win rate | Undertraining | Increase `--num-iterations` |
| RuntimeError: shape mismatch | Bug in code | Run `python validate.py` |
| Slow training on CPU | Expected | Use `--device cuda` or reduce model size |

## Key Takeaways

1. **Causal attention** enforces temporal causality
2. **Belief state** compresses history into latent vector
3. **Learned dynamics** implicitly models opponents
4. **Multi-head loss** trains all components jointly
5. **Self-play** generates training data automatically
6. **Modular design** enables research extensions

---

**For detailed info, see: README.md | ARCHITECTURE.md | ROADMAP.md**
