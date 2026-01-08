"""
Main training loop for Poker Transformer Agent.

Combines:
1. Self-play game generation
2. Search-based target creation
3. Model training with multi-head loss
4. Periodic evaluation and checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import logging

from src.config.config import ExperimentConfig
from src.model import PokerTransformerAgent
from src.environment import KuhnPoker
from src.training.search import SelfPlayBuffer, run_self_play_game, LatentSpaceSearcher
from src.evaluation import visualize_training_summary, PokerEvaluator


class PokerTrainer:
    """
    End-to-end trainer for Poker Transformer Agent.
    
    Loss function:
    L = λ_π * KL(π_target || π_θ) 
      + λ_v * (V_target - V_θ)²
      + λ_d * ||z_{t+1} - g_θ(z_t, a_t)||²
      + [λ_opp * opponent_range_loss]
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create environment and model
        self.env = KuhnPoker(
            initial_stack=config.environment.initial_stack,
            ante=config.environment.ante,
            max_raises=config.environment.max_raises,
            seed=config.environment.seed,
        )
        
        self.agent = PokerTransformerAgent(config).to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.training.num_iterations
        )
        
        # Experience buffer
        self.buffer = SelfPlayBuffer(max_size=10000)
        
        # Optional: Search engine
        self.searcher = None
        if config.training.search_type == "mcts":
            self.searcher = LatentSpaceSearcher(
                self.agent, self.env,
                num_simulations=config.training.num_simulations,
                rollout_depth=config.training.rollout_depth,
            )
        
        # Logging
        self.log_dir = Path(config.log_dir) / config.name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._create_logger()
        self.metrics = {
            'iteration': [],
            'game_reward': [],
            'policy_loss': [],
            'value_loss': [],
            'transition_loss': [],
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.training.optimizer == "adam":
            return optim.Adam(
                self.agent.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "sgd":
            return optim.SGD(
                self.agent.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_logger(self) -> logging.Logger:
        """Create logger for training."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_dir / "training.log")
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training with config:\n{self.config.to_dict()}")
        
        for iteration in range(self.config.training.num_iterations):
            # 1. Self-play games
            self.logger.info(f"Iteration {iteration}: Running self-play...")
            games = self._run_self_play_batch(self.config.training.games_per_iteration)
            
            # Record average reward
            avg_reward = np.mean([g.rewards[0] for g in games])
            self.metrics['game_reward'].append(avg_reward)
            self.logger.info(f"  Average reward (P0): {avg_reward:.3f}")
            
            # 2. Train on collected experiences
            self.logger.info(f"  Training model...")
            loss_dict = self._train_on_batch(games)
            
            # 3. Update metrics
            for key, value in loss_dict.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
            
            self.metrics['iteration'].append(iteration)

            # 4. Periodic evaluation (exploitability, etc.)
            if iteration % max(1, self.config.training.checkpoint_freq // 2) == 0:
                try:
                    evaluator = PokerEvaluator(self.agent, self.config, self.log_dir)
                    eval_results = evaluator.run_full_evaluation()

                    # Log exploitability
                    if 'exploitability' in eval_results and eval_results['exploitability'] is not None:
                        exploitability = eval_results['exploitability']
                        self.metrics['exploitability'] = self.metrics.get('exploitability', [])
                        self.metrics['exploitability'].append(exploitability)
                        self.logger.info(f"  Exploitability: {exploitability:.6f}")

                except Exception as e:
                    self.logger.warning(f"Evaluation failed: {e}")

            # 5. Logging
            self.logger.info(
                f"  Losses - Policy: {loss_dict.get('policy_loss', 0):.4f}, "
                f"Value: {loss_dict.get('value_loss', 0):.4f}, "
                f"Transition: {loss_dict.get('transition_loss', 0):.4f}"
            )

            # 6. Checkpointing
            if iteration % self.config.training.checkpoint_freq == 0:
                self._save_checkpoint(iteration)
            
            # 6. Learning rate update
            self.scheduler.step()
        
        self.logger.info("Training complete!")
        self._save_metrics()
        
        # Generate visualizations
        self.logger.info("Generating visualizations...")
        try:
            visualize_training_summary(self.log_dir)
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
    
    def _run_self_play_batch(self, num_games: int) -> List:
        """Run batch of self-play games."""
        games = []
        for _ in range(num_games):
            game = run_self_play_game(self.agent, self.env, self.searcher)
            games.append(game)
            self.buffer.add_game(game)
        return games
    
    def _train_on_batch(self, games: List) -> Dict[str, float]:
        """
        Train model on batch of games.
        
        Returns:
            loss_dict: Dict with individual loss components
        """
        self.agent.train()
        
        batch_size = self.config.training.batch_size
        total_losses = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'transition_loss': 0.0,
            'opponent_range_loss': 0.0,
        }
        num_batches = 0
        
        # Process games in batches
        for i in range(0, len(games), batch_size):
            batch_games = games[i:i+batch_size]
            
            # Create batch tensors
            batch_data = self._prepare_batch(batch_games)
            
            if batch_data is None:
                continue
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.agent(batch_data['observations'])
            
            # Compute losses
            loss_dict = self._compute_loss(outputs, batch_data)
            total_loss = sum(loss_dict.values())
            
            # Backward and optimize
            total_loss.backward()
            
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(),
                    self.config.training.grad_clip,
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                total_losses[key] += value.item()
            
            num_batches += 1
        
        # Average losses
        if num_batches > 0:
            for key in total_losses:
                total_losses[key] /= num_batches
        
        return total_losses
    
    def _prepare_batch(self, games: List) -> Optional[Dict]:
        """Prepare batch of games for training."""
        observations = []
        policy_targets = []
        value_targets = []
        actions = []
        next_observations = []
        
        for game in games:
            if not game.observations:
                continue
            
            observations.extend(game.observations)
            
            # If no search values, use actual payoffs as targets
            if game.search_values:
                value_targets.extend(game.search_values)
            else:
                # Simple baseline: repeat final payoff
                payoff = game.rewards.get(0, 0)
                value_targets.extend([payoff] * len(game.observations))
            
            # Policy targets
            if game.search_policies:
                policy_targets.extend(game.search_policies)
            else:
                # Uniform policy as default
                policy_targets.extend([np.ones(4) / 4.0] * len(game.observations))
            
            # Actions and next observations for transition learning
            if hasattr(game, 'actions') and game.actions:
                actions.extend(game.actions)
                # Create next observations by shifting: next_obs[i] = obs[i+1]
                # For the last action, we don't have a next observation, so skip it
                if len(game.observations) > len(game.actions):
                    # Normal case: more obs than actions (final obs has no action)
                    next_observations.extend(game.observations[1:len(game.actions)+1])
                else:
                    # Fallback: observations and actions same length
                    next_observations.extend(game.observations[1:])
        
        if not observations:
            return None
        
        batch = {
            'observations': observations,
            'policy_targets': torch.tensor(np.stack(policy_targets), 
                                          dtype=torch.float32, device=self.device),
            'value_targets': torch.tensor(np.array(value_targets), 
                                         dtype=torch.float32, device=self.device),
        }
        
        # Add transition targets if available
        if actions and next_observations:
            batch['actions'] = torch.tensor(np.array(actions), 
                                           dtype=torch.long, device=self.device)
            batch['next_observations'] = next_observations
        
        return batch
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                      batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute training loss with all components."""
        losses = {}
        weights = self.config.training.loss_weights
        
        # 1. Policy loss: KL divergence
        policy_logits = outputs['policy_logits']
        policy_probs = F.softmax(policy_logits, dim=-1)
        policy_targets = batch['policy_targets']
        
        kl_loss = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            policy_targets,
            reduction='batchmean',
        )
        losses['policy_loss'] = weights['policy'] * kl_loss
        
        # 2. Value loss: MSE
        values = outputs['values'].squeeze(-1)
        value_targets = batch['value_targets']
        
        value_loss = F.mse_loss(values, value_targets)
        losses['value_loss'] = weights['value'] * value_loss
        
        # 3. Transition loss: L2 distance between predicted and actual next beliefs
        if 'actions' in batch and 'next_observations' in batch and len(batch['next_observations']) > 0:
            # Get current beliefs
            current_beliefs = outputs['belief_states']
            actions_tensor = batch['actions']
            
            # Verify shapes match
            if len(batch['next_observations']) != current_beliefs.shape[0]:
                # If mismatch, adjust actions to match
                actions_tensor = actions_tensor[:len(batch['next_observations'])]
                current_beliefs = current_beliefs[:len(batch['next_observations'])]
            
            # Predict next beliefs using transition model
            predicted_next_beliefs = self.agent.predict_next_belief(current_beliefs, actions_tensor)
            
            # Encode actual next observations to get target next beliefs
            actual_next_beliefs_list = []
            for next_obs in batch['next_observations']:
                # Encode the next observation (wrap in list since agent expects list)
                with torch.no_grad():
                    next_output = self.agent([next_obs])
                    actual_next_beliefs_list.append(next_output['belief_states'])
            
            if actual_next_beliefs_list:
                actual_next_beliefs = torch.cat(actual_next_beliefs_list, dim=0)
                # Ensure shapes match
                if actual_next_beliefs.shape == predicted_next_beliefs.shape:
                    transition_loss = F.mse_loss(predicted_next_beliefs, actual_next_beliefs)
                    losses['transition_loss'] = weights['transition'] * transition_loss
                else:
                    losses['transition_loss'] = torch.tensor(0.0, device=self.device)
            else:
                losses['transition_loss'] = torch.tensor(0.0, device=self.device)
        else:
            losses['transition_loss'] = torch.tensor(0.0, device=self.device)
        
        # 4. Opponent range loss: optional
        losses['opponent_range_loss'] = torch.tensor(0.0, device=self.device)
        
        return losses
    
    def _save_checkpoint(self, iteration: int):
        """Save model checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'agent_state': self.agent.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config.__dict__,
        }
        
        checkpoint_path = self.log_dir / f"checkpoint_iter{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_metrics(self):
        """Save training metrics."""
        metrics_path = self.log_dir / "metrics.json"
        
        # Convert numpy arrays to lists
        metrics_serializable = {}
        for key, values in self.metrics.items():
            metrics_serializable[key] = [float(v) for v in values]
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        self.logger.info(f"Saved metrics to {metrics_path}")
