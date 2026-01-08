# Architecture & Design Document

## System Overview

This is a research-grade Poker AI built on the principle:

> **"The Transformer approximates a belief-state MDP. The transition model learns opponent dynamics. The value head approximates counterfactual EV."**

### Data Flow

```
[Observable History] 
    cards: int
    actions: [(player, action, amount)]
         │
         ↓
[BeliefStateTransformer]
    - Embed cards, actions, amounts
    - Stack into sequence
    - Causal multi-head attention
    - Positional encoding
         │
         ↓
    z_t: Latent Belief State (dense vector)
         │
    ┌────┴────┬──────────┬──────────────┐
    │          │          │              │
    ↓          ↓          ↓              ↓
 ValueHead  PolicyHead  TransitionModel OpponentRange
   V(z)       π(a|z)     g(z,a)→z'      P(card|z)
```

## Key Components

### 1. Belief State Encoder (`src/model/transformer.py`)

**Why Transformers?**
- Handles variable-length sequences (from 0 to N actions)
- Attention learns which parts of history matter
- Causal masking respects temporal causality
- Scales to longer games (e.g., Leduc with 2+ streets)

**Architecture:**
```
Input: (sequence_length, input_dim)
  ├─ Card embedding:     (1, d_card)
  ├─ Action embedding:   (1, d_action)
  └─ Bet embedding:      (1, d_bet)
  = concatenated: (1, d_model)

├─ Positional Encoding (sine/cosine)
├─ [Transformer Block] × num_layers
│   ├─ Causal Self-Attention (masked)
│   │   ├─ Q = linear(x)
│   │   ├─ K = linear(x)
│   │   ├─ V = linear(x)
│   │   ├─ scores = Q @ K.T / sqrt(d_head)
│   │   ├─ mask: triu(1) = future invisible
│   │   └─ attn = softmax(scores + mask)
│   ├─ ResNet: x + Attn(x)
│   ├─ LayerNorm
│   ├─ FFN: Linear → ReLU → Linear
│   └─ ResNet: x + FFN(x)
│
└─ Output: z_t (last non-padded token)
```

**Key Design Decisions:**
- **Causal masking**: `triu(1)` masks future positions
- **Padding handling**: Track non-padded length, extract final token
- **Separate embeddings**: Cards, actions, amounts have independent embedding spaces
- **Residual connections**: Improve gradient flow through deep stacks

### 2. Transition Model (`src/model/heads.py: LatentTransitionModel`)

**What it learns:**
- How beliefs update: $z_{t+1} = g_\theta(z_t, a_t)$
- Implicitly: opponent's reaction to actions
- Implicitly: how much information actions reveal

**Architecture:**
```
Input: [z_t; one_hot(action)]
  (concat latent dim + num_actions)

├─ Linear(in, hidden) → ReLU
├─ Linear(hidden, hidden) → ReLU
└─ Linear(hidden, latent_dim)

Output: z_{t+1}
```

**Why deterministic (for now)?**
- Simpler to train (no divergence penalties)
- Still captures average opponent behavior
- Can extend to probabilistic (z_mean, z_logvar)

### 3. Value Head (`src/model/heads.py: ValueHead`)

**What it estimates:**
$$V_\theta(z_t) \approx \mathbb{E}[\text{payoff} \mid z_t]$$

This is the counterfactual EV: "If I reach this belief state, what's my expected value?"

**Architecture:**
```
Input: z_t (latent_dim,)

├─ Linear(latent_dim, hidden) → ReLU
├─ Linear(hidden, hidden) → ReLU
└─ Linear(hidden, 1)

Output: scalar value
```

**Training Target:**
- From MCTS: bootstrapped value (value + discounted tree value)
- From rollouts: Monte Carlo return
- Simple: final payoff (works in Kuhn, but less efficient)

### 4. Policy Head (`src/model/heads.py: PolicyHead`)

**What it outputs:**
$$\pi_\theta(a \mid z_t) = \text{softmax}(\text{logits})$$

This is the action selection distribution.

**Architecture:**
```
Input: z_t (latent_dim,)

├─ Linear(latent_dim, hidden) → ReLU
├─ Linear(hidden, hidden) → ReLU
└─ Linear(hidden, 4)  # [FOLD, CALL, RAISE, CHECK]

Output: (4,) logits → softmax → probabilities
```

**Action Masking:**
```python
# Before softmax
illegal_mask = -inf where action is illegal
logits = logits + illegal_mask
probs = softmax(logits)  # Illegal actions → 0
```

**Training Target:**
- From MCTS: visit counts → probabilities
- From improved policy: rollout-based improvement
- Loss: `KL(π_target || π_θ)` = cross-entropy + entropy

### 5. Opponent Range Predictor (Optional) (`src/model/heads.py: OpponentRangePredictor`)

**Interpretability tool:** Can we extract opponent's inferred hand from attention?

**Use cases:**
- Verify model learns hand strength correlation
- Detect exploitable patterns
- Regularization: encourage belief encoding

**Architecture:** Similar to ValueHead, outputs logits over 3 cards.

## Training Loop

### Algorithm

```
for iteration = 1 to N:
    # Phase 1: Self-Play Collection
    for game = 1 to B:
        play_game(Agent, environment)  # Get trajectory + payoff
        [Optional] run_search(Agent, game_state)  # Get improved targets

    # Phase 2: Experience Processing
    for each game trajectory:
        create_training_examples(observations, actions, payoffs, search_targets)

    # Phase 3: Model Training
    for batch in shuffle(all_examples):
        # Forward pass
        beliefs = agent.encode_belief(observations)
        values = agent.value_head(beliefs)
        logits = agent.policy_head(beliefs)
        next_beliefs = agent.transition_model(beliefs, actions)

        # Compute losses
        L_policy = KL(π* || π_θ)       # Search targets
        L_value = MSE(V* - V_θ)        # Bootstrapped targets
        L_trans = MSE(z'_actual - z'_pred)  # If available

        # Total loss
        L = λ_π · L_policy + λ_v · L_value + λ_d · L_trans

        # Optimize
        optimizer.backward(L)
        gradient_clip()
        optimizer.step()

    # Phase 4: Evaluation & Checkpoint
    if iteration % checkpoint_freq == 0:
        evaluate(agent)
        save_checkpoint(agent, iteration)
```

### Loss Components

1. **Policy Loss** (KL divergence):
   $$L_\pi = \text{KL}(\pi^* \| \pi_\theta) = \sum_a \pi^*(a) \log \frac{\pi^*(a)}{\pi_\theta(a)}$$
   - Trains agent to match search-improved policy
   - High entropy → exploration
   - Low entropy → exploitation

2. **Value Loss** (MSE):
   $$L_v = \| V^* - V_\theta(z_t) \|^2$$
   - Bootstrapped target: $V^* = r + \gamma V(z_{t+1})$
   - Or Monte Carlo: $V^* = G_t$ (discounted return)
   - Enables credit assignment

3. **Transition Loss** (MSE, optional):
   $$L_d = \| z'_{\text{actual}} - g_\theta(z_t, a_t) \|^2$$
   - Requires encoding both $z_t$ and $z_{t+1}$
   - Regularizes belief dynamics to stay consistent

4. **Opponent Range Loss** (cross-entropy, optional):
   $$L_{\text{opp}} = -\sum_c \mathbb{1}[\text{opp\_card} = c] \log P(c|z_t)$$
   - Encourages learning of hand distributions
   - Interpretability regularizer

### Weight Configuration

```python
loss_weights = {
    'policy': 1.0,           # Always on
    'value': 1.0,            # Always on
    'transition': 0.5,       # Optional, for consistency
    'opponent_range': 0.1,   # Optional, for interpretability
}
```

**Tuning:**
- Increase `policy` if agent explores too much
- Increase `value` if value estimates are noisy
- Increase `transition` if beliefs become incoherent
- Increase `opponent_range` to force learning explicit hand distributions

## Search in Latent Space

### MCTS (`src/training/search.py: LatentSpaceSearcher`)

**Why search?**
- Raw self-play is on-policy; search generates improved targets
- Allows extrapolation beyond training distribution
- Provides supervision for value + policy

**Algorithm (simplified):**
```
for simulation = 1 to M:
    # Start at root belief
    z = encode_belief(current_observation)

    # Simulate forward
    for step = 1 to depth:
        if terminal: return payoff
        
        # Sample action from policy
        a ~ π_θ(·|z)
        
        # Update belief via transition
        z' = g_θ(z, a)
        z ← z'

    # Bootstrap value
    v = V_θ(z)
    return v

# Aggregate visits
visit_counts[a] ∝ #simulations(a)
search_policy = visit_counts / sum(visit_counts)
search_value = avg(all_rollout_values)
```

**Limitations in current code:**
- Simplified: only random rollouts in belief space
- Real: would integrate environment stepping + trajectory weighting
- Extension: alpha-zero-style value+policy combined search

### Rollout (Alternative)

```
for rollout = 1 to R:
    sample_trajectory_from_policy()
    return final_payoff

rollout_value = mean(payoffs)
```

Faster but less stable than MCTS.

## Belief State Geometry

### What is the latent space?

The latent dimension is not interpretable a priori. It encodes:

1. **Card information** (which cards are possible)
2. **Opponent modeling** (inferred from betting)
3. **Game progress** (street, pot odds)
4. **Uncertainty** (continuous representation)

### Analysis Tools (`src/model/agent.py: BeliefStateGeometry`)

1. **Attention Analysis**
   ```python
   opponent_attention = geometry.get_attention_to_opponent_actions(...)
   # Do higher card cards get more attention?
   ```

2. **Variance Decomposition**
   ```python
   var_per_dim = geometry.belief_state_variance(beliefs)
   # Which dimensions encode information?
   ```

3. **Attention Flow**
   ```python
   flow = geometry.attention_flow_analysis(attention_weights)
   # How does information flow through layers?
   ```

### Visualization Suggestions

- **t-SNE/UMAP**: Project belief states to 2D, color by actual outcome
- **Attention heatmaps**: Show which positions attend to which
- **Value landscape**: Heatmap of V(z) over belief space samples
- **Card correlation**: Does attention to opponent's actions correlate with card strength?

## Extensibility

### Adding to Kuhn → Leduc

```python
# 1. Extend environment
class LeduckPoker(KuhnPoker):
    DECK = list(range(6))  # 6 cards
    STREETS = 2            # Preflop + Flop
    
# 2. Update config
config.environment.game_type = 'leduc'
config.model.max_sequence_length = 256  # More history

# 3. Increase model capacity
config.model.latent_dim = 128
config.model.num_layers = 6

# 4. More training
config.training.num_iterations = 1000
```

### Adding Probabilistic Transitions

```python
class ProbabilisticTransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        self.mean_net = MLP(...)
        self.logvar_net = MLP(...)
    
    def forward(self, z, action):
        mean = self.mean_net([z, one_hot(action)])
        logvar = self.logvar_net([z, one_hot(action)])
        
        # Reparameterization trick
        eps = torch.randn_like(mean)
        z_next = mean + exp(0.5 * logvar) * eps
        
        return z_next, mean, logvar
    
    def kl_loss(self, mean, logvar):
        return -0.5 * (1 + logvar - mean**2 - exp(logvar)).sum()
```

### Adding Belief-Based Exploration

```python
# Entropy bonus on policy
entropy_bonus = -sum(π * log(π))
action_logits = policy_head(z) + β * entropy_bonus
```

### Adding Imitation Pre-training

```python
# 1. Generate expert games with CFR solver
expert_games = cfr_solver.generate_games(N=10000)

# 2. Imitation loss
def imitation_loss(policy_logits, expert_policy):
    return KL(expert_policy || softmax(policy_logits))

# 3. Pre-train before self-play
for batch in expert_games:
    optimize(imitation_loss)
```

## Testing & Validation

### Checklist

- [x] Environment game plays correctly to terminal
- [x] Model instantiates without errors
- [x] Forward pass produces correct shapes
- [x] Training loss decreases over time
- [x] Evaluation metrics compute
- [x] Checkpoints save/load correctly
- [x] Belief states change with actions
- [x] Value function shows reasonable outputs
- [x] Policy produces valid action distributions
- [x] Attention weights sum to 1.0

### Debug Checklist

If model isn't learning:
- [ ] Check loss is decreasing in `metrics.json`
- [ ] Verify agent rewards improving with `evaluate_vs_random()`
- [ ] Inspect belief variance: should increase over training
- [ ] Check attention entropy: should be stable, not collapsing
- [ ] Verify data loading: sample batch and inspect
- [ ] Gradient flow: check backward pass computes gradients
- [ ] Learning rate: try higher/lower values

## References & Further Reading

1. **MuZero** (Schaal & Silver et al., 2020)
   - Learned environment models in latent space
   - Credit assignment via bootstrapped returns

2. **AlphaZero** (Silver et al., 2017)
   - Self-play + search for policy/value targets
   - MCTS in environment space (we do latent space)

3. **Transformer Models** (Vaswani et al., 2017)
   - Attention mechanisms for sequences
   - Causal masking for autoregressive generation

4. **CFR** (Zinkevich et al., 2008)
   - Nash equilibrium in imperfect info games
   - Regret minimization

5. **POMDPs** (Kaelbling et al., 1998)
   - Belief-state representations
   - Planning under partial observability

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Status**: Research-Grade
