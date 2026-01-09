# Interactive Poker Visualization Guide

This guide explains the Pygame-based real-time visualization of the Poker Transformer Agent.

## Overview

The visualization displays the internal neural network activations as the agent plays poker. It helps you understand:

- How the Transformer encodes game history into belief states
- How attention weights are distributed across the action sequence
- How the value function estimates expected payoffs
- How the transition model predicts belief updates
- How these representations evolve during a single hand

## Layout

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  Game Info          │  Cards                     │  Controls                 │
│  - Status           │  - Opponent (hidden)      │  - R: Reset               │
│  - Step count       │  - Your card (visible)    │  - A: Auto-play           │
│  - Final payoff     │                            │  - Q: Quit                │
├─────────────────────┴────────────────────────────┴──────────────────────────┤
│  Action History    │  Pot Info  │  Belief State  │  Value  │  Transition  │
│  - P0: CHECK       │  - Pot     │  (64D)         │  - EV    │  - Error     │
│  - P1: RAISE       │  - Stacks  │  [||||||||]    │  - Graph │  [||||||||]  │
│  - P0: CALL        │            │                │         │              │
├────────────────────┴────────────┴────────────────┴─────────┴──────────────┤
│                      Attention Weights Heatmap                            │
│                      (Shows attention to history)                          │
└──────────────────────────────────────────────────────────────────────────────┘
                    [FOLD] [CALL] [RAISE] [CHECK]  (Action buttons)
```

## Panel Descriptions

### 1. Game Info (Top Left)
- **Status**: Current game state (IN PROGRESS / OVER)
- **Step**: Number of actions taken so far
- **Final Payoff**: Net chips won/lost when game ends
- **Mode**: AUTO (watch agent play) or MANUAL (choose actions)

### 2. Cards (Top Center)
- **Opponent Card**: Hidden (blue back) until showdown
- **Your Card**: Face-up showing J, Q, or K
- Card rank order: K (highest) > Q > J (lowest)

### 3. Controls (Top Right)
- **R**: Reset and start a new game
- **A**: Toggle between auto-play and manual mode
- **Space**: Take one step forward (in auto mode)
- **Q**: Quit visualization

### 4. Action History (Middle Left)
Shows the sequence of actions in the current hand:
- **P0**: Player 0 (you)
- **P1**: Opponent
- **FOLD**: Surrender the pot
- **CALL**: Match opponent's bet
- **RAISE**: Increase the bet
- **CHECK**: Pass action without betting

Colors indicate action type for quick scanning.

### 5. Pot Info (Middle)
- **Pot**: Total chips in the middle
- **Player 0/1**: Each player's remaining stack
- Pot grows when players CALL or RAISE

### 6. Belief State (Middle Center)
Shows the **64-dimensional latent vector** that the Transformer learned to encode the entire game history.

**What it means:**
- Each bar is one dimension of the belief state
- **Red bars**: Positive activation
- **Blue bars**: Negative activation
- Height = activation magnitude
- The vector captures: your card, opponent's likely range, betting dynamics, game progress

**What to watch for:**
- How beliefs change after each action
- Which dimensions activate strongly (what they represent is learned)
- Pattern differences between winning and losing games

### 7. Value Function (Middle Right)
Shows the **expected payoff (EV)** predicted by the value head.

**What it means:**
- Positive = agent expects to win chips
- Negative = agent expects to lose chips
- Magnitude = confidence in the estimate

**Graph:**
- Shows value history throughout the game
- X-axis: step number
- Y-axis: normalized value (min to max seen)

**What to watch for:**
- How value changes after opponent actions
- Whether value correlates with actual outcomes
- How confident the agent is in different positions

### 8. Transition Model (Far Right)
Shows the **prediction error** of the transition model.

**What it means:**
- The transition model predicts: `z_next = g_theta(z_current, action)`
- Error = difference between predicted and actual next belief
- Bars = error magnitude per dimension
- Color intensity = size of error (green = low, red = high)

**What to watch for:**
- Low error = model accurately predicts how beliefs change
- High error = surprise / model learning opportunity
- Which dimensions are hardest to predict

### 9. Attention Weights (Bottom)
Shows the **multi-head attention** from the Transformer's final layer.

**What it means:**
- Heatmap cells = how much attention is paid to each position
- Rows = query positions (later in sequence)
- Columns = key positions (earlier in sequence)
- **Blue** = low attention, **Red** = high attention

**Semantic interpretation:**
- Column 0: Your own card (always important)
- Later columns: Opponent actions (important for inferring their range)

**What to watch for:**
- Does the agent attend more to opponent raises?
- Does attention shift as the game progresses?
- Do different heads specialize (e.g., one for cards, one for bets)?

### 10. Action Buttons (Bottom)
Only visible in **manual mode**:
- **FOLD**: Give up (lose current pot contribution)
- **CALL**: Match opponent's bet
- **RAISE**: Bet more chips
- **CHECK**: Pass without betting (only if no one has bet yet)

Grayed-out buttons = illegal actions given current game state.

## Game Rules (Kuhn Poker)

1. **Cards**: 3 cards in deck (J, Q, K), 2 players each get 1 card
2. **Ante**: Both players post 1 chip blind
3. **Betting**:
   - Player 0 acts first
   - Can CHECK or RAISE
   - If opponent bets, can FOLD, CALL, or RAISE
4. **Showdown**:
   - Game ends when both players CHECK, or one FOLDs, or both CALL same amount
   - High card wins the pot
5. **Payoff**:
   - Winner takes entire pot
   - Your net = chips won - chips lost

## Interpreting Activations

### Belief State Patterns

**When you have a King (strongest card):**
- Belief dimensions should indicate high confidence
- Value should be positive
- Attention might focus less on opponent actions (you're ahead)

**When you have a Jack (weakest card):**
- Belief dimensions should indicate weakness
- Value should be negative
- Attention might focus heavily on opponent's bets (are they bluffing?)

**After opponent raises:**
- Belief should update to reflect stronger opponent range
- Value should decrease (they probably have a good hand)
- Attention to that RAISE should be high

**After you raise:**
- Belief should incorporate your aggression
- Value might increase if you're representing strength
- Attention to your own card should be consistent

### Value Function Learning

**Early training:**
- Value predictions might be random or poorly correlated with outcomes
- Large swings between steps

**After training:**
- Value should correlate with:
  - Your card strength (higher card = higher value)
  - Pot size (more at stake = more variance)
  - Betting history (aggressive opponent = lower value)
- Should predict actual payoffs accurately

### Attention Learning

**Initial state:**
- Attention might be noisy or uniform

**After training:**
- **Row 0 (your card)**: Always high attention (primary info source)
- **Opponent raises**: High attention (critical for range inference)
- **Your checks**: Lower attention (less informative)
- **Recent actions**: Higher attention than older ones (recency bias)

### Transition Model Errors

**High error:**
- Model is surprised by belief changes
- Could indicate:
  - Complex game state transitions
  - Model still learning dynamics
  - Rare/edge case scenarios

**Low error:**
- Model accurately predicts how beliefs evolve
- Indicates good understanding of game dynamics
- Belief space is smooth and predictable

## Usage Examples

### Analyze a Single Hand

```bash
# Manual mode - take each action yourself
python visualize.py
```

Watch how:
- Your card influences initial belief and value
- Each opponent action shifts the belief
- Value estimates change as pot and information change
- Attention focuses on informative actions

### Watch Trained Agent Play

```bash
# Auto-play mode with trained model
python visualize.py --model logs/poker_transformer_default/checkpoint.pth --auto
```

Compare:
- How agent plays with different cards
- Betting patterns across hands
- Consistency of value predictions
- Attention patterns across different situations

### Debug Model Behavior

```bash
# Slow auto-play to analyze each step
python visualize.py --auto --play-delay 3000
```

Look for:
- Unexpected value drops (did agent misevaluate?)
- High transition errors (model confusion?)
- Unusual attention patterns (what's it focusing on?)
- Belief dimension activations (what do different dims encode?)

## Advanced Analysis

### Compare Random vs Trained Model

1. Run visualization with random model:
   ```bash
   python visualize.py --auto
   ```
   Note: Values might be random, attention noisy

2. Run visualization with trained model:
   ```bash
   python visualize.py --model logs/.../checkpoint.pth --auto
   ```
   Note: Structured patterns emerge

### Track a Single Belief Dimension

Pick a dimension (e.g., dimension 0) and watch it across:
- Different starting cards
- Different action sequences
- Multiple games

Hypothesis: Some dimensions encode:
- Card strength
- Pot size
- Game stage
- Opponent aggression

### Correlation Analysis

Run 20-30 hands in auto-mode and record:
- Final value vs actual outcome
- Attention to opponent raises vs opponent card
- Transition error vs game complexity

Look for patterns that reveal what the model learned.

## Troubleshooting

### Window too large/small
```bash
python visualize.py --width 1280 --height 800
```

### Auto-play too fast/slow
```bash
python visualize.py --auto --play-delay 500   # Fast (500ms)
python visualize.py --auto --play-delay 3000  # Slow (3s)
```

### Model file not found
Ensure you've trained a model:
```bash
python main.py --num-iterations 10
python visualize.py --model logs/poker_transformer_default/checkpoint.pth
```

### Game doesn't end
If games get stuck (rare), press `R` to reset.

## Technical Details

### Belief State Encoding
```
Observable State:
  - Own card (J/Q/K)
  - Action history: [(player, action, amount), ...]
  - Current player
  - Stacks, pot, street

Transformer Encoder:
  - Embed card → 16 dims
  - Embed action → 16 dims
  - Embed bet amount → 16 dims
  - Concatenate → 48 dims per token
  - 3 layers of causal attention
  - Output: 64-dim belief vector
```

### Value Function
```
Belief (64D) → MLP (128, 128, 1) → Scalar EV
```

### Transition Model
```
[Belief (64D), Action One-Hot (4D)] → MLP (128, 128, 64) → Next Belief
```

### Attention
```
Layer 3, Head 0: (sequence_length × sequence_length) attention weights
Shows how each position attends to all previous positions
```

## Further Reading

- [Architecture Deep Dive](../documentations/ARCHITECTURE.md)
- [Analysis Report](../documentations/ANALYSIS_REPORT.md)
- [Main README](../README.md)
