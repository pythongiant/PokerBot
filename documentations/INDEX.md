# 📖 Documentation Index

Welcome to the Poker Transformer project! Use this guide to navigate the documentation.

## 🚀 Start Here

**New to the project?** Follow this path:

1. **[README.md](README.md)** (5 min read)
   - What is this project?
   - High-level architecture
   - Key features & objectives

2. **[GETTING_STARTED.md](GETTING_STARTED.md)** (10 min read)
   - Installation instructions
   - Quick validation (`quickstart.py`)
   - Running your first training
   - Common questions (FAQ)

3. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** (5 min reference)
   - Visual architecture diagrams
   - Common commands
   - Hyperparameter tuning
   - Error troubleshooting

4. **Run it!** (15 minutes)
   ```bash
   python quickstart.py
   python main.py --num-iterations 50 --eval
   ```

---

## 📚 Deep Learning

**Want to understand the system deeply?**

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** (30 min read)
   - Detailed component breakdown
   - Mathematical formulations
   - Training algorithm pseudocode
   - Belief state geometry
   - Extensibility guide

2. **Source Code** (Code review)
   - Start: `src/model/agent.py` (unified model)
   - Then: `src/model/transformer.py` (encoder)
   - Then: `src/model/heads.py` (prediction heads)
   - Then: `src/training/trainer.py` (training loop)

3. **Key Concepts**
   - Causal attention (no future peeking)
   - Belief states (latent representations)
   - Learned transition dynamics
   - Value function (credit assignment)
   - Self-play (data generation)

---

## � Visualization & Analysis

**Want to understand what the model learned?**

1. **[VISUALIZATIONS.md](VISUALIZATIONS.md)** (15 min read)
   - Belief state projections (2D plots)
   - Value landscapes
   - Attention heatmaps
   - Training metrics dashboard
   - Interpretation guide

2. **Automatic Visualizations**
   - Generated after each training run
   - Saved to `logs/<experiment>/visualizations/`
   - No manual code needed!

3. **Manual Generation**
   ```python
   from src.evaluation import BeliefStateVisualizer
   visualizer = BeliefStateVisualizer(agent, config)
   visualizer.generate_belief_report(num_games=50)
   ```

**Planning to extend or publish?**

1. **[ROADMAP.md](ROADMAP.md)** (20 min read)
   - Phase 1-5 research plan
   - 20+ concrete extension ideas
   - Implementation difficulty levels
   - Impact assessments

2. **[examples/examples.py](examples/examples.py)** (runnable)
   - 6 example workflows
   - Ablation studies
   - Custom training loops
   - Belief analysis

3. **Research Topics**
   - Kuhn → Leduc poker extension
   - Bet sizing (continuous actions)
   - Multi-player variants
   - Exact exploitability
   - Meta-learning / few-shot

---

## 🛠️ Getting Help

### Quick Lookup Tables

**📋 By File Purpose**
| Need | File | Purpose |
|------|------|---------|
| Train a model | `main.py` | Entry point with CLI args |
| Understand architecture | `src/model/agent.py` | Unified model |
| Game rules | `src/environment/kuhn.py` | Kuhn poker logic |
| Training code | `src/training/trainer.py` | Main training loop |
| Config | `src/config/config.py` | All hyperparameters |
| Evaluation | `src/evaluation/evaluator.py` | Head-to-head, metrics |

**🎯 By Task**
| Task | Document | File |
|------|----------|------|
| Start training | GETTING_STARTED | main.py |
| Understand model | ARCHITECTURE | src/model/agent.py |
| Run ablation | ROADMAP | examples/examples.py |
| Debug issue | QUICK_REFERENCE | Troubleshooting section |
| Extend system | ARCHITECTURE | Extension section |
| Cite work | IMPLEMENTATION_SUMMARY | Citation format |

**🔍 By Concept**
| Concept | Document | Section |
|---------|----------|---------|
| Transformer | ARCHITECTURE | "Belief State Encoder" |
| Causal attention | QUICK_REFERENCE | "Architecture (Visual)" |
| Training loss | ARCHITECTURE | "Loss Components" |
| Belief geometry | ARCHITECTURE | "Belief State Geometry" |
| Self-play | IMPLEMENTATION_SUMMARY | "Training Workflow" |
| Search | ARCHITECTURE | "Search in Latent Space" |

---

## 📂 File Structure Quick Guide

```
poker_bot/
├── README.md                    ← START HERE (overview)
├── GETTING_STARTED.md           ← Installation & quick start
├── ARCHITECTURE.md              ← Deep technical dive
├── ROADMAP.md                   ← Research extensions
├── QUICK_REFERENCE.md           ← Commands & troubleshooting
├── IMPLEMENTATION_SUMMARY.md    ← This project in summary
│
├── main.py                      ← Entry point: python main.py --help
├── quickstart.py                ← Validation: python quickstart.py
├── validate.py                  ← Component tests: python validate.py
│
└── src/                         ← Source code
    ├── config/config.py         ← All hyperparameters
    ├── environment/kuhn.py      ← Game logic
    ├── model/agent.py           ← Unified model
    ├── model/transformer.py     ← Belief encoder
    ├── model/heads.py           ← Value, policy, etc.
    ├── training/trainer.py      ← Training loop
    ├── training/search.py       ← Self-play & search
    └── evaluation/evaluator.py  ← Evaluation utilities
```

---

## 🎓 Learning Path by Background

### For RL Researchers
1. README.md (what's new?)
2. ARCHITECTURE.md (how does it work?)
3. ROADMAP.md (what's next?)
4. Code: src/training/trainer.py
5. Code: src/model/agent.py

### For Game Theory Researchers
1. README.md (overview)
2. QUICK_REFERENCE.md (visual guide)
3. ARCHITECTURE.md (game-theoretic principles)
4. ROADMAP.md (exploitability section)
5. Code: src/environment/kuhn.py

### For ML/Transformer Experts
1. README.md (what's novel?)
2. ARCHITECTURE.md ("Belief State Encoder" section)
3. Code: src/model/transformer.py
4. Code: src/model/heads.py
5. ROADMAP.md (extensions)

### For Industry Practitioners
1. GETTING_STARTED.md (how to use?)
2. QUICK_REFERENCE.md (commands & tuning)
3. examples/examples.py (runnable examples)
4. README.md (background)
5. ROADMAP.md (what's possible?)

---

## 💬 FAQ / Common Questions

**Q: Where do I start?**  
A: Read [GETTING_STARTED.md](GETTING_STARTED.md), then run `python quickstart.py`

**Q: How do I train a model?**  
A: `python main.py --num-iterations 100 --eval`

**Q: How do I understand the code?**  
A: Read [ARCHITECTURE.md](ARCHITECTURE.md) then browse `src/`

**Q: What are good hyperparameters?**  
A: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) "Hyperparameter Tuning"

**Q: How do I debug training?**  
A: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) "Error Handling"

**Q: How do I extend this system?**  
A: See [ROADMAP.md](ROADMAP.md) for 20+ ideas with difficulty levels

**Q: Can I use this in research?**  
A: Yes! See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) "Citation"

**Q: What's the performance?**  
A: See README.md "Success Criteria" and run evaluation

---

## 📊 Documentation Statistics

| Document | Type | Size | Read Time |
|----------|------|------|-----------|
| README.md | Overview | Long | 15 min |
| GETTING_STARTED.md | Guide | Long | 10 min |
| ARCHITECTURE.md | Technical | Very Long | 30 min |
| ROADMAP.md | Research | Long | 20 min |
| QUICK_REFERENCE.md | Reference | Medium | 5 min |
| IMPLEMENTATION_SUMMARY.md | Summary | Medium | 10 min |
| VISUALIZATIONS.md | Guide | Medium | 15 min |

**Total**: ~105 minutes of documentation  
**Code**: ~2800 lines of research-grade Python

---

## 🔗 Cross-References

### Mentions of Key Concepts

**Belief State**
- README.md: "The Transformer approximates a belief-state MDP"
- ARCHITECTURE.md: Section "Belief State Encoder"
- QUICK_REFERENCE.md: Data structures section

**Causal Attention**
- ARCHITECTURE.md: "Causal Attention"
- QUICK_REFERENCE.md: Visual diagrams
- Code: src/model/transformer.py

**Training Loop**
- README.md: "Training Loop" section
- ARCHITECTURE.md: "Training Loop" section
- IMPLEMENTATION_SUMMARY.md: "Training Workflow"
- Code: src/training/trainer.py

**Evaluation**
- README.md: "Experiments to Support"
- ARCHITECTURE.md: "Testing & Validation"
- Code: src/evaluation/evaluator.py

**Extensions**
- ROADMAP.md: Phases 2-5
- ARCHITECTURE.md: "Extensibility" section
- examples/examples.py: Working code

---

## 🚦 Reading Recommendations by Goal

### Goal: Get it running (15 min)
→ GETTING_STARTED.md + `python quickstart.py`

### Goal: Understand the system (1 hour)
→ README.md → ARCHITECTURE.md → Code review

### Goal: Run experiments (30 min)
→ QUICK_REFERENCE.md + examples/examples.py

### Goal: Extend/research (2 hours)
→ ROADMAP.md → ARCHITECTURE.md Extensions → Code

### Goal: Publish (1 day)
→ All docs → Code → Experiments → ROADMAP phase ideas

---

## 📝 Document Maintenance

**Last Updated**: January 2024  
**Version**: 1.0  
**Status**: Research-Grade, Active  

**Documentation Quality**:
- [x] All major components documented
- [x] Code comments explain "why"
- [x] Examples runnable
- [x] Cross-references working
- [x] Consistent formatting
- [x] Math rendered in ARCHITECTURE.md

---

## 🎯 Next Steps

1. **Just starting?**
   ```bash
   # Read first
   cat README.md
   cat GETTING_STARTED.md
   
   # Then run
   python quickstart.py
   ```

2. **Want to train?**
   ```bash
   python main.py --num-iterations 100 --eval
   ```

3. **Want to understand?**
   ```bash
   # Read ARCHITECTURE.md, then:
   code src/model/agent.py
   ```

4. **Want to research?**
   ```bash
   # Read ROADMAP.md and pick a project
   cat ROADMAP.md
   ```

---

**Choose your path above and happy hacking! 🚀**

If you have questions, check the relevant document or run `python validate.py` to test components.
