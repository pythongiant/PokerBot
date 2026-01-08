# Research Roadmap & Future Extensions

This document outlines potential research directions and extensions to the Poker Transformer framework.

## Phase 1: Foundation (Current)
- [x] Kuhn poker environment
- [x] Causal Transformer belief encoder
- [x] Latent transition model (deterministic)
- [x] Value and policy heads
- [x] Basic self-play training
- [x] Evaluation utilities

## Phase 2: Enhanced Learning (Near-term)

### 2.1 Improved Search Mechanisms
**Goal**: Move from random rollouts to principled MCTS

**Implementation**:
```python
class ImprovedMCTS(LatentSpaceSearcher):
    """
    Full MCTS with:
    - UCB-based exploration (visited nodes prioritized)
    - Transposition tables (memoize belief states)
    - Alpha-Zero style combined value+policy search
    """
    def __init__(self, ...):
        self.node_table = {}  # z_hash -> statistics
        self.c_puct = 1.0    # Exploration constant
    
    def select_action(self, z, legal_actions):
        # UCB: select_arg = argmax(Q + c*P*sqrt(N)/(1+n))
        return best_action
    
    def backup(self, value, path):
        # Propagate value back through all nodes
        for node in reversed(path):
            node.update(value)
```

**Impact**: Better convergence, fewer simulations needed

**Effort**: Medium (requires environment integration)

### 2.2 Probabilistic Transition Model
**Goal**: Represent uncertainty in belief dynamics

**Implementation**:
```python
class StochasticTransitionModel(nn.Module):
    def forward(self, z, a):
        mean = self.mean_net([z, a])
        logvar = self.logvar_net([z, a])
        
        # Reparameterization trick
        z_next = mean + exp(0.5*logvar) * randn()
        return z_next, mean, logvar

# Training: add KL regularization
kl_loss = -0.5 * (1 + logvar - mean**2 - exp(logvar)).sum()
```

**Impact**: Explicit uncertainty modeling, better calibration

**Effort**: Low (modular change to transition_model.py)

### 2.3 Opponent Modeling
**Goal**: Explicitly extract and track opponent strategies

**Implementation**:
```python
class OpponentModel(nn.Module):
    """
    Separate model of opponent's strategy.
    - Predicts opponent's next action
    - Updated from observations
    """
    def forward(self, z):
        # Predict opponent's action distribution
        return opponent_policy  # shape: (num_actions,)

# In training loop:
# Compare predicted vs actual opponent action
opponent_model_loss = KL(predicted_policy || observed_action)
```

**Impact**: Interpretability, better exploitation

**Effort**: Medium (requires opponent action tracking)

### 2.4 Mixed Precision Training
**Goal**: Faster training, reduced memory

**Implementation**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = agent(batch)
    loss = compute_loss(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Impact**: 2-3x speedup on GPU, 20% memory savings

**Effort**: Low (framework integration)

## Phase 3: Game Extensions (Medium-term)

### 3.1 Leduc Poker
**Goal**: Two-player, two-street poker with 6-card deck

**Key differences from Kuhn**:
- More decisions (flop + turn)
- Public community card
- Larger game tree
- Richer strategy space

**Implementation checklist**:
- [ ] `environment/leduc.py`: Full game logic
- [ ] Update observation encoding for public cards
- [ ] Increase model capacity:
  ```python
  latent_dim = 128
  num_layers = 6
  max_sequence_length = 256
  ```
- [ ] Train longer (1000+ iterations)
- [ ] Benchmark vs CFR baseline

**Effort**: Medium (1-2 weeks)

**Expected challenges**:
- Larger game tree → slower self-play
- More complex strategies → harder to learn
- Need better search (MCTS essential)

### 3.2 Bet Sizing
**Goal**: Continuous action space instead of discrete

**Implementation**:
```python
class ContinuousPolicyHead(nn.Module):
    def forward(self, z):
        mean_size = self.mean_net(z)           # [0, 1] via sigmoid
        logvar_size = self.logvar_net(z)       # log variance
        
        # Reparameterization trick
        eps = randn()
        bet_size = mean_size + exp(0.5*logvar_size) * eps
        
        # Clamp to valid range
        bet_size = clamp(bet_size, min_bet, stack_size)
        
        return bet_size

# Loss: NLL of bet size
loss = -log_prob_beta_distribution(actual_bet, mean_size, logvar_size)
```

**Impact**: More realistic game, harder opponent to exploit

**Effort**: Low (change policy head only)

### 3.3 Multi-Player (3+ players)
**Goal**: More complex game theory (fewer dominant strategies)

**Key challenges**:
- Exponentially larger game tree
- Need separate belief states for each opponent
- Coalition formation possible

**Implementation**:
```python
class MultiPlayerAgent(nn.Module):
    def __init__(self, num_players):
        # Separate belief encoder for each opponent
        self.belief_encoders = nn.ModuleList([
            BeliefStateTransformer(...) 
            for _ in range(num_players - 1)
        ])
        # Combined value/policy
        self.combined_value = ValueHead(latent_dim * (num_players-1))
```

**Impact**: Richer game-theoretic interactions

**Effort**: High (complex environment, significant model changes)

## Phase 4: Theory & Analysis (Longer-term)

### 4.1 Exact Exploitability Computation
**Goal**: Quantify agent strength vs. optimal play

**Implementation**:
```python
def compute_exploitability(agent, game_tree):
    """
    Use game-theoretic solver (nashpy, gambit) to compute:
    best_response_value = max_strategy E[payoff against agent]
    exploitability = opponent_value - agent_value
    """
    # Requires enumerating game tree for exact solution
    # Feasible for Kuhn, intractable for larger games
```

**Tools needed**:
- `nashpy`: Pure Python game theory
- `gambit`: Full game solver (requires C++ build)

**Impact**: Ground truth evaluation, research publication

**Effort**: Low (integration with external library)

### 4.2 Belief State Visualization
**Goal**: Understand learned representations

**Implementation**:
```python
def visualize_belief_landscape():
    """
    1. Sample random belief states
    2. Evaluate V(z) and π(z) on 2D grid
    3. Plot heatmaps, contours
    4. Overlay actual game outcomes
    """
    # Use t-SNE or UMAP to project high-dim space to 2D
    import umap
    
    beliefs = collect_beliefs_from_games()  # (N, latent_dim)
    reducer = umap.UMAP(n_components=2)
    beliefs_2d = reducer.fit_transform(beliefs)
    
    # Evaluate value/policy at each point
    values = agent.predict_value(beliefs)
    
    # Plot
    scatter(beliefs_2d, c=values, cmap='RdYlGn')
```

**Impact**: Interpretability paper, visual understanding

**Effort**: Medium (requires visualization library)

### 4.3 Attention Analysis as Opponent Modeling
**Goal**: Extract explicit opponent ranges from attention patterns

**Implementation**:
```python
def extract_opponent_range(agent, observation, top_k=3):
    """
    1. Forward pass, extract attention weights
    2. Find which positions attention focuses on
    3. Map back to actions → infer opponent's likely cards
    4. Return as explicit probability distribution
    """
    outputs = agent(observation)
    attn_weights = outputs['attention_weights'][-1]  # Last layer
    
    # Average over heads, get final position's attention
    final_attn = attn_weights[0, :, -1, :].mean(dim=0)  # (seq_len,)
    
    # Find most-attended opponent action positions
    opponent_positions = [i for i, (p, _, _) in enumerate(action_history) if p == 1]
    opponent_attention = final_attn[opponent_positions]
    
    # Map to inferred hand strength
    return opponent_attention
```

**Impact**: Interpretable exploitability, theory grounding

**Effort**: Medium (requires careful attention bookkeeping)

### 4.4 Imitation Learning from CFR
**Goal**: Combine learned dynamics with game-theoretic baselines

**Implementation**:
```python
class HybridAgent(nn.Module):
    """
    Train on:
    1. CFR-generated expert games (supervised)
    2. Self-play games (RL)
    
    Benefits:
    - Faster convergence (pre-training)
    - Better exploration (CFR baseline)
    - Hybrid training signal
    """
    def train_phase(self, stage):
        if stage == 'imitation':
            loss = imitation_loss
        elif stage == 'selfplay':
            loss = rl_loss
        elif stage == 'mixed':
            loss = imitation_loss + rl_loss
```

**Impact**: Faster convergence, better convergence guarantees

**Effort**: Medium (curriculum learning strategy)

## Phase 5: Advanced Research Topics (Speculative)

### 5.1 Hierarchical RL
**Goal**: Learn multi-scale strategies (high-level + low-level)

```python
class HierarchicalAgent(nn.Module):
    def __init__(self):
        # High-level policy: fold / play
        self.high_policy = PolicyHead(latent_dim, num_actions=2)
        
        # Low-level policies: conditional on high action
        self.low_policies = nn.ModuleDict({
            'fold': PolicyHead(...),
            'play': PolicyHead(...),
        })
```

**References**: Feudal Networks (Vezhnevets et al.), Options Framework (Sutton et al.)

### 5.2 Meta-Learning
**Goal**: Quickly adapt to opponent types

```python
class MetaLearner(nn.Module):
    """
    Learn a learning algorithm that adapts to new opponents.
    MAML-style or gradient-based adaptation.
    """
    def adapt(self, opponent_games, num_steps=5):
        # Few-shot learning: observe opponent, adapt policy
        for _ in range(num_steps):
            loss = compute_loss(opponent_games)
            loss.backward()
            self.meta_update()
```

### 5.3 Game-Theoretic Regularization
**Goal**: Ensure learned agent respects game-theoretic principles

```python
def game_theoretic_loss(agent, payoffs):
    """
    Minimize exploitability via regret bounds:
    - Regret: sum of (benchmark - actual payoff)
    - Loss: encourage low-regret strategies
    """
    regret = compute_cfr_regret(agent, game_tree)
    return regret
```

### 5.4 Transfer Learning
**Goal**: Pre-train on Kuhn, fine-tune on Leduc

```python
def transfer_learning():
    # 1. Train on Kuhn until convergence
    agent_kuhn = train_agent('kuhn')
    
    # 2. Initialize Leduc agent with Kuhn weights
    agent_leduc = LeduckAgent()
    agent_leduc.belief_encoder = agent_kuhn.belief_encoder  # Transfer
    
    # 3. Fine-tune on Leduc
    fine_tune(agent_leduc, 'leduc', num_iterations=500)
```

## Implementation Priorities

### High Priority (Next Quarter)
1. MCTS integration → ~2 weeks, high impact
2. Probabilistic transitions → ~1 week, low risk
3. Leduc poker → ~2 weeks, significant extension
4. Exploitability metrics → ~1 week, necessary for research

### Medium Priority (Next 6 Months)
1. Opponent modeling (explicit)
2. Belief visualization
3. Attention analysis
4. Mixed precision training

### Lower Priority (Future)
1. Multi-player extension
2. Hierarchical RL
3. Meta-learning
4. Full game solver integration

## Contribution Guidelines

When implementing extensions:

1. **Modularity**: Keep changes isolated (e.g., new file, new class)
2. **Tests**: Add validation for new components
3. **Documentation**: Update README & ARCHITECTURE
4. **Reproducibility**: Save configs, log hyperparameters
5. **Ablations**: Include experimental results showing impact

## References for Future Work

- Schaal & Silver et al. (2020): MuZero
- Vezhnevets et al. (2017): Feudal Networks
- Finn et al. (2017): MAML
- Brown & Sandholm (2018): Libratus (poker bot)
- Burch et al. (2014): Solving Imperfect Information Games
- Zinkevich et al. (2008): Regret Minimization in Games

---

**Last Updated**: 2024  
**Status**: Active Research
