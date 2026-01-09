import pygame
import numpy as np
import torch
from typing import List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PokerTransformerAgent
from src.environment import KuhnPoker, Action, ObservableState
from src.config import DEFAULT_CONFIG


class PygamePokerVisualizer:
    def __init__(self, agent: PokerTransformerAgent, config, width=1400, height=900, human_vs_model=False, human_player=0, auto_continue=True, alternate=False, random_ai=False):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))

        self.agent = agent
        self.config = config
        self.env = KuhnPoker(initial_stack=100, ante=1, seed=42)
        self.random_ai = random_ai  # Whether to use random policy for AI

        # Human vs Model mode
        self.human_vs_model = human_vs_model
        self.human_player = human_player  # 0 or 1
        self.model_player = 1 - human_player
        self.auto_continue = auto_continue  # Whether to auto-continue games
        self.alternate = alternate  # Whether to alternate who starts each game

        # Track total payoffs across games
        self.total_payoffs = [0.0, 0.0]  # Cumulative payoffs for each player
        self.games_completed = 0

        if human_vs_model:
            pygame.display.set_caption("Poker: Human vs AI")
        else:
            pygame.display.set_caption("Poker Transformer Agent Visualization")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        self.card_names = ['J', 'Q', 'K']
        self.action_names = ['FOLD', 'CALL', 'RAISE', 'CHECK']
        self.action_colors = {
            'FOLD': (255, 100, 100),
            'CALL': (100, 255, 100),
            'RAISE': (255, 200, 100),
            'CHECK': (100, 200, 255)
        }
        
        self.colors = {
            'bg': (30, 30, 35),
            'panel': (45, 45, 50),
            'text': (220, 220, 220),
            'accent': (100, 150, 200),
            'card_back': (50, 100, 150),
            'card_front': (240, 240, 240),
            'card_text': (20, 20, 20),
            'button': (70, 120, 170),
            'button_hover': (90, 140, 190),
            'pot': (255, 200, 50),
        }
        
        self.current_step = 0
        self.belief_history = []
        self.value_history = []
        self.transition_history = []
        self.attention_weights_history = []
        
        self.game_running = False
        self.game_over = False
        self.auto_play = False
        self.play_delay = 1500
        self.games_played = 0
        self.max_games = 10  # Play 10 games in sequence for longer visualization
        
        self.buttons = []
        self.selected_action = None
        self.last_update_time = 0
        
        # Store initial cards to prevent them from changing during game
        self.player_card = None
        self.opponent_card = None
        
        self.setup_ui()
    
    def setup_ui(self):
        button_y = self.height - 70
        button_width = 120
        button_height = 40
        button_spacing = 15
        start_x = (self.width - (4 * button_width + 3 * button_spacing)) // 2
        
        for i, action in enumerate(['FOLD', 'CALL', 'RAISE', 'CHECK']):
            rect = pygame.Rect(
                start_x + i * (button_width + button_spacing),
                button_y,
                button_width,
                button_height
            )
            self.buttons.append({
                'action': action,
                'rect': rect,
                'hover': False
            })
    
    def reset_game(self):
        # Update total payoffs from previous game
        if self.game_over and self.game_state.payoffs:
            for player_id in [0,1]:
                self.total_payoffs[player_id] += self.game_state.payoffs[player_id]
            self.games_completed +=1

        # If alternating, swap players
        if self.human_vs_model and self.alternate and self.games_completed > 0:
            self.human_player, self.model_player = self.model_player, self.human_player

        self.game_state, self.obs = self.env.reset()
        self.current_step = 0
        self.belief_history = []
        self.value_history = []
        self.transition_history = []
        self.attention_weights_history = []
        self.game_over = False
        self.selected_action = None
        self.last_update_time = pygame.time.get_ticks()

        # Store initial cards for consistent visualization
        self.player_card = self.obs.own_card
        self.opponent_card = self.game_state.private_cards[1]

        self.agent.eval()
        with torch.no_grad():
            self.current_belief, self.attention_info = self.agent.encode_belief([self.obs])
            self.current_value = self.agent.predict_value(self.current_belief)[0, 0].item()

            self.belief_history.append(self.current_belief[0].cpu().numpy())
            self.value_history.append(self.current_value)
            self.attention_weights_history.append([w.clone() for w in self.attention_info['attention_weights']])
    
    def get_legal_actions(self):
        return self.env.get_legal_actions(self.game_state.current_player)
    
    def step_game(self, action: Action):
        if self.game_over:
            return
        
        player_id = self.game_state.current_player
        
        with torch.no_grad():
            next_belief_pred = self.agent.predict_next_belief(
                self.current_belief, 
                torch.tensor([action.value], device=self.current_belief.device)
            )
            self.transition_history.append(next_belief_pred[0].cpu().numpy())
        
        # Calculate proper betting amounts based on action type
        if action == Action.CHECK:
            step_amount = 0
        elif action == Action.CALL:
            # Let the environment calculate the proper call amount automatically
            step_amount = None
        elif action == Action.RAISE:
            # For RAISE, let environment calculate default (call + 1 chip)
            step_amount = None
        else:
            step_amount = 0
            
        self.game_state, self.obs, is_terminal = self.env.step(player_id, action, step_amount)
        
        with torch.no_grad():
            if not is_terminal:
                self.current_belief, self.attention_info = self.agent.encode_belief([self.obs])
                self.current_value = self.agent.predict_value(self.current_belief)[0, 0].item()
            else:
                final_belief, _ = self.agent.encode_belief([self.obs])
                self.current_belief = final_belief
                self.current_value = self.agent.predict_value(final_belief)[0, 0].item()
            
            self.belief_history.append(self.current_belief[0].cpu().numpy())
            self.value_history.append(self.current_value)
            if hasattr(self, 'attention_info') and 'attention_weights' in self.attention_info:
                self.attention_weights_history.append([w.clone() for w in self.attention_info['attention_weights']])
        
        self.current_step += 1
        
        if is_terminal:
            self.game_over = True
    
    def model_step(self):
        """Let the model make its move if it's the model's turn"""
        if self.game_over:
            # In human vs model mode, continue with new games automatically if auto_continue is enabled
            # In regular mode, only continue if auto_play is enabled
            should_continue = (self.human_vs_model and self.auto_continue) or (self.auto_play and self.games_played < self.max_games)
            if should_continue:
                pygame.time.delay(2000)  # Longer delay for human to see result
                self.games_played += 1
                self.reset_game()
            return

        current_player = self.game_state.current_player

        # In human vs model mode, only let model play its turn
        if self.human_vs_model and current_player == self.human_player:
            return  # Wait for human input

        # In auto-play mode or when it's model's turn, make a move
        current_time = pygame.time.get_ticks()
        if current_time - self.last_update_time > self.play_delay:
            legal_actions = self.get_legal_actions()
            if len(legal_actions) == 0:
                action = Action.CHECK  # fallback
            else:
                if self.random_ai:
                    # Use random action from legal actions
                    import random
                    action = random.choice(legal_actions)
                else:
                    # Use trained model
                    with torch.no_grad():
                        policy_logits = self.agent.predict_policy(self.current_belief)[0]  # (4,)
                        # Create mask for legal actions
                        legal_mask = torch.zeros(4, device=policy_logits.device)
                        for a in legal_actions:
                            legal_mask[a.value] = 1.0
                        # Mask illegal actions with large negative
                        masked_logits = policy_logits + (1 - legal_mask) * (-1e9)
                        action_idx = torch.argmax(masked_logits).item()
                        action = Action(action_idx)
            self.step_game(action)
            self.last_update_time = current_time
    
    def draw_panel(self, x, y, width, height, title=None):
        pygame.draw.rect(self.screen, self.colors['panel'], (x, y, width, height))
        pygame.draw.rect(self.screen, self.colors['accent'], (x, y, width, height), 2)
        
        if title:
            title_surf = self.font_medium.render(title, True, self.colors['text'])
            title_rect = title_surf.get_rect(center=(x + width // 2, y + 20))
            self.screen.blit(title_surf, title_rect)
        
        return y + 40
    
    def draw_card(self, x, y, width, height, card_idx, face_up=True, player_name=""):
        if face_up:
            pygame.draw.rect(self.screen, self.colors['card_front'], (x, y, width, height))
            pygame.draw.rect(self.screen, (0, 0, 0), (x, y, width, height), 3)
            
            card_name = self.card_names[card_idx]
            card_surf = self.font_large.render(card_name, True, self.colors['card_text'])
            card_rect = card_surf.get_rect(center=(x + width // 2, y + height // 2))
            self.screen.blit(card_surf, card_rect)
        else:
            pygame.draw.rect(self.screen, self.colors['card_back'], (x, y, width, height))
            pygame.draw.rect(self.screen, (100, 140, 180), (x, y, width, height), 3)
            pygame.draw.circle(self.screen, (70, 110, 150), (x + width // 2, y + height // 2), 20)
        
        if player_name:
            name_surf = self.font_small.render(player_name, True, self.colors['text'])
            name_rect = name_surf.get_rect(center=(x + width // 2, y + height + 15))
            self.screen.blit(name_surf, name_rect)
    
    def draw_action_history(self):
        panel_x, panel_y = 50, 350
        panel_width = 250
        panel_height = 180
        
        y = self.draw_panel(panel_x, panel_y, panel_width, panel_height, "Action History")
        
        history = self.obs.action_history
        for i, (player_id, action, amount) in enumerate(history):
            if self.human_vs_model:
                player_name = "Player" if player_id == self.human_player else "Model"
            else:
                player_name = "P0" if player_id == 0 else "P1"
            action_name = action.name
            color = self.action_colors[action_name]

            text = f"{player_name}: {action_name}"
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (panel_x + 15, y + i * 25))
    
    def draw_pot_info(self):
        panel_x, panel_y = 320, 350
        panel_width = 180
        panel_height = 180
        
        y = self.draw_panel(panel_x, panel_y, panel_width, panel_height, "Pot Info")
        
        pot_text = f"Pot: {self.game_state.pot}"
        surf = self.font_large.render(pot_text, True, self.colors['pot'])
        self.screen.blit(surf, (panel_x + panel_width // 2 - surf.get_width() // 2, y + 20))
        
        for player_id, stack in enumerate(self.game_state.stacks):
            if self.human_vs_model:
                player_name = "Player" if player_id == self.human_player else "Model"
            else:
                player_name = f"Player {player_id}"
            text = f"{player_name}: {stack}"
            surf = self.font_medium.render(text, True, self.colors['text'])
            self.screen.blit(surf, (panel_x + 20, y + 70 + player_id * 35))
    
    def draw_belief_state(self):
        panel_x, panel_y = 520, 350
        panel_width = 300
        panel_height = 250

        y = self.draw_panel(panel_x, panel_y, panel_width, panel_height, "Belief State (3D Grid)")

        if len(self.belief_history) > 0:
            belief = self.belief_history[-1]

            # Reshape to 8x8 grid for 3D-like visualization
            grid_size = 8
            belief_grid = belief[:grid_size*grid_size].reshape(grid_size, grid_size)

            cell_width = max(1, (panel_width - 40) // grid_size)
            cell_height = max(1, (panel_height - 80) // grid_size)

            max_val = np.max(np.abs(belief_grid)) + 0.1

            for i in range(grid_size):
                for j in range(grid_size):
                    val = belief_grid[i, j]
                    norm_val = val / max_val
                    intensity = int(255 * abs(norm_val))
                    if norm_val > 0:
                        color = (intensity, 100, 100)
                    else:
                        color = (100, 100, intensity)

                    cell_x = panel_x + 20 + j * cell_width
                    cell_y = y + 10 + i * cell_height

                    pygame.draw.rect(self.screen, color, (cell_x, cell_y, cell_width, cell_height))
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_x, cell_y, cell_width, cell_height), 1)

            stats_text = f"Mean: {np.mean(belief):.3f} | Std: {np.std(belief):.3f}"
            stats_surf = self.font_small.render(stats_text, True, self.colors['text'])
            self.screen.blit(stats_surf, (panel_x + 10, y + grid_size * cell_height + 20))
    
    def draw_value_function(self):
        panel_x, panel_y = 840, 350
        panel_width = 220
        panel_height = 180
        
        y = self.draw_panel(panel_x, panel_y, panel_width, panel_height, "Value Function")
        
        if len(self.value_history) > 0:
            current_value = self.value_history[-1]
            
            color = (100, 255, 100) if current_value >= 0 else (255, 100, 100)
            value_text = f"{current_value:.4f}"
            value_surf = self.font_large.render(value_text, True, color)
            value_rect = value_surf.get_rect(center=(panel_x + panel_width // 2, y + 40))
            self.screen.blit(value_surf, value_rect)
            
            if len(self.value_history) > 1:
                graph_height = panel_height - 100
                graph_width = panel_width - 40
                
                values = np.array(self.value_history)
                min_val, max_val = np.min(values), np.max(values)
                if max_val - min_val > 0:
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(values)
                
                for i in range(len(values) - 1):
                    x1 = panel_x + 20 + i * (graph_width / max(1, len(values) - 1))
                    y1 = panel_y + panel_height - 20 - normalized[i] * graph_height
                    x2 = panel_x + 20 + (i + 1) * (graph_width / max(1, len(values) - 1))
                    y2 = panel_y + panel_height - 20 - normalized[i + 1] * graph_height
                    
                    pygame.draw.line(self.screen, self.colors['accent'], (x1, y1), (x2, y2), 2)
            
            min_text = f"Min: {np.min(self.value_history):.4f}"
            max_text = f"Max: {np.max(self.value_history):.4f}"
            min_surf = self.font_small.render(min_text, True, self.colors['text'])
            max_surf = self.font_small.render(max_text, True, self.colors['text'])
            self.screen.blit(min_surf, (panel_x + 10, y + 80))
            self.screen.blit(max_surf, (panel_x + 10, y + 100))
    
    def draw_transition_model(self):
        panel_x, panel_y = 1080, 350
        panel_width = 240
        panel_height = 250
        
        y = self.draw_panel(panel_x, panel_y, panel_width, panel_height, "Transition Model")
        
        if len(self.transition_history) > 0:
            predicted = self.transition_history[-1]
            if len(self.belief_history) > 0:
                actual = self.belief_history[-1] if not self.game_over else self.belief_history[-1]
                
                diff = np.abs(predicted - actual)
                error = np.mean(diff)
                
                error_text = f"Error: {error:.4f}"
                error_surf = self.font_medium.render(error_text, True, self.colors['text'])
                self.screen.blit(error_surf, (panel_x + 10, y + 10))
                
                bar_width = max(1, (panel_width - 40) // min(len(diff), 64))
                max_diff = np.max(diff) + 0.01
                max_bar_height = panel_height - 80
                
                for i in range(min(len(diff), 64)):
                    norm_diff = diff[i] / max_diff
                    bar_height = max(1, int(norm_diff * max_bar_height))
                    
                    intensity = int(255 * norm_diff)
                    color = (intensity, 255 - intensity, 100)
                    
                    bar_x = int(panel_x + 20 + i * bar_width)
                    bar_y = int(panel_y + panel_height - 20 - bar_height)
                    
                    pygame.draw.rect(self.screen, color, (bar_x, bar_y, max(1, bar_width - 1), bar_height))
        else:
            no_data_surf = self.font_small.render("No transitions yet", True, self.colors['text'])
            self.screen.blit(no_data_surf, (panel_x + 10, y + 40))
    
    def draw_attention_heatmap(self):
        panel_x, panel_y = 50, 550
        panel_width = 450
        panel_height = 250
        
        y = self.draw_panel(panel_x, panel_y, panel_width, panel_height, "Attention Weights")
        
        if len(self.attention_weights_history) > 0:
            attention = self.attention_weights_history[-1][-1][0, 0].cpu().numpy()
            
            heatmap_height = int(panel_height - 80)
            heatmap_width = int(panel_width - 40)
            
            cell_width = max(1, heatmap_width // max(1, attention.shape[0]))
            cell_height = max(1, heatmap_height // max(1, attention.shape[1]))
            
            max_attn = np.max(attention) + 0.01
            
            for i in range(attention.shape[0]):
                for j in range(attention.shape[1]):
                    norm_attn = attention[i, j] / max_attn
                    color_val = int(255 * norm_attn)
                    color = (color_val, 100, 255 - color_val)
                    
                    cell_x = int(panel_x + 20 + j * cell_width)
                    cell_y = int(panel_y + 40 + i * cell_height)
                    
                    pygame.draw.rect(self.screen, color, (cell_x, cell_y, cell_width, cell_height))
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_x, cell_y, cell_width, cell_height), 1)
            
            legend_text = "Low (Blue) -> High (Red)"
            legend_surf = self.font_small.render(legend_text, True, self.colors['text'])
            self.screen.blit(legend_surf, (panel_x + 10, panel_y + panel_height - 25))
        else:
            no_data_surf = self.font_small.render("No attention data yet", True, self.colors['text'])
            self.screen.blit(no_data_surf, (panel_x + 10, y + 40))
    
    def draw_game_info(self):
        panel_x, panel_y = 50, 150
        panel_width = 250
        panel_height = 160
        
        y = self.draw_panel(panel_x, panel_y, panel_width, panel_height, "Game Info")
        
        status_text = "Status: " + ("OVER" if self.game_over else "IN PROGRESS")
        status_color = (255, 100, 100) if self.game_over else (100, 255, 100)
        status_surf = self.font_medium.render(status_text, True, status_color)
        self.screen.blit(status_surf, (panel_x + 20, y))
        
        step_text = f"Step: {self.current_step} | Game: {self.games_played + 1}/{self.games_completed + 1}"
        step_surf = self.font_small.render(step_text, True, self.colors['text'])
        self.screen.blit(step_surf, (panel_x + 20, y + 30))

        # Show total payoffs (accumulated across all games)
        if self.human_vs_model:
            human_total = self.total_payoffs[self.human_player]
            model_total = self.total_payoffs[self.model_player]
            total_text = f"Total: You: {human_total:+.1f} | AI: {model_total:+.1f}"
            total_color = (100, 255, 100) if human_total >= 0 else (255, 100, 100)
            total_surf = self.font_small.render(total_text, True, total_color)
            self.screen.blit(total_surf, (panel_x + 20, y + 50))

        if self.game_over:
            if self.human_vs_model:
                # Show human player's payoff prominently
                human_payoff = self.game_state.payoffs[self.human_player] if self.game_state.payoffs else 0
                payoff_text = f"This Hand: {human_payoff:+.1f}"
                payoff_color = (100, 255, 100) if human_payoff >= 0 else (255, 100, 100)

                # Also show who won
                if human_payoff > 0:
                    result_text = "You Won!"
                elif human_payoff < 0:
                    result_text = "You Lost"
                else:
                    result_text = "Tie Game"
                result_surf = self.font_medium.render(result_text, True, payoff_color)
                result_rect = result_surf.get_rect(center=(panel_x + panel_width // 2, y + 70))
                self.screen.blit(result_surf, result_rect)
            else:
                # Regular mode - show player 0's payoff
                payoff = self.game_state.payoffs[0] if self.game_state.payoffs else 0
                payoff_text = f"Final Payoff: {payoff:.1f}"
                payoff_color = (100, 255, 100) if payoff >= 0 else (255, 100, 100)

            payoff_surf = self.font_large.render(payoff_text, True, payoff_color)
            payoff_rect = payoff_surf.get_rect(center=(panel_x + panel_width // 2, y + 100))
            self.screen.blit(payoff_surf, payoff_rect)
        
        if self.human_vs_model:
            mode_text = f"Mode: HUMAN VS AI (You: P{self.human_player})"
        else:
            mode_text = f"Mode: {'AUTO' if self.auto_play else 'MANUAL'}"
        mode_surf = self.font_small.render(mode_text, True, self.colors['accent'])
        self.screen.blit(mode_surf, (panel_x + 20, y + 130))
    
    def draw_cards(self):
        panel_x, panel_y = 320, 50
        panel_width = 600
        panel_height = 160

        y = self.draw_panel(panel_x, panel_y, panel_width, panel_height, "Cards")

        card_width = 100
        card_height = 140

        # Use stored cards to ensure they don't change during the game
        if self.player_card is not None and self.opponent_card is not None:
            if self.human_vs_model:
                # In human vs model mode, show who's who
                if self.human_player == 0:
                    left_name = "Model"
                    right_name = "Player"
                else:
                    left_name = "Player"
                    right_name = "Model"
            else:
                left_name = "Opponent"
                right_name = "Your Card"

            # Left card (opponent)
            left_card_x = panel_x + 50
            left_card_y = y + 10
            self.draw_card(left_card_x, left_card_y, card_width, card_height,
                            self.opponent_card, face_up=True, player_name=left_name)

            # Right card (player)
            right_card_x = panel_x + panel_width - 50 - card_width
            right_card_y = y + 10
            self.draw_card(right_card_x, right_card_y, card_width, card_height,
                          self.player_card, face_up=True, player_name=right_name)
    
    def draw_buttons(self):
        legal_actions = self.get_legal_actions() if not self.game_over else []

        mouse_pos = pygame.mouse.get_pos()

        for button in self.buttons:
            action = button['action']
            rect = button['rect']
            
            is_legal = any(a.name == action for a in legal_actions)
            
            button['hover'] = rect.collidepoint(mouse_pos)
            
            if is_legal and button['hover']:
                color = self.colors['button_hover']
            elif is_legal:
                color = self.colors['button']
            else:
                color = (80, 80, 80)
            
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, (100, 100, 100), rect, 2, border_radius=8)
            
            text_surf = self.font_medium.render(action, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)
    
    def draw_controls(self):
        panel_x, panel_y = 950, 150
        panel_width = 240
        panel_height = 160

        y = self.draw_panel(panel_x, panel_y, panel_width, panel_height, "Controls")

        if self.human_vs_model:
            if self.game_over:
                controls = [
                    "R - New Game",
                    "Q - Quit"
                ]
                if self.auto_continue:
                    turn_text = "Next game starting..."
                    turn_color = (100, 200, 255)
                else:
                    turn_text = "Press R for new game"
                    turn_color = (255, 200, 100)
            else:
                controls = [
                    "R - Reset Game",
                    "Q - Quit",
                    "Mouse - Your Turn"
                ]
                # Show whose turn it is
                current_player = self.game_state.current_player
                if current_player == self.human_player:
                    turn_text = "Your Turn!"
                    turn_color = (100, 255, 100)
                else:
                    turn_text = "AI Thinking..."
                    turn_color = (255, 200, 100)
        else:
            controls = [
                "R - Reset Game",
                "A - Toggle Auto Play",
                "Space - Single Step (Auto)",
                "Q - Quit"
            ]
            if self.auto_play:
                turn_text = "Auto Play: ON"
                turn_color = (100, 255, 100)
            else:
                turn_text = "Auto Play: OFF"
                turn_color = (255, 100, 100)

        for i, text in enumerate(controls):
            surf = self.font_small.render(text, True, self.colors['text'])
            self.screen.blit(surf, (panel_x + 15, y + i * 30))

        turn_surf = self.font_medium.render(turn_text, True, turn_color)
        self.screen.blit(turn_surf, (panel_x + 15, y + 130))
    
    def draw(self):
        self.screen.fill(self.colors['bg'])
        
        self.draw_game_info()
        self.draw_cards()
        self.draw_action_history()
        self.draw_pot_info()
        self.draw_belief_state()
        self.draw_value_function()
        self.draw_transition_model()
        self.draw_attention_heatmap()
        self.draw_controls()
        
        # Show buttons when it's human's turn (or in manual mode), but not when game is over
        should_show_buttons = False
        if self.human_vs_model:
            should_show_buttons = not self.game_over and self.game_state.current_player == self.human_player
        else:
            should_show_buttons = not self.auto_play and not self.game_over

        if should_show_buttons:
            self.draw_buttons()
        
        pygame.display.flip()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_r:
                    self.games_played = 0
                    self.reset_game()
                elif event.key == pygame.K_a:
                    self.auto_play = not self.auto_play
                elif event.key == pygame.K_SPACE and self.auto_play:
                    self.model_step()
            
            # Allow mouse clicks in human vs model mode when it's human's turn, or in manual mode
            allow_mouse_clicks = False
            if self.human_vs_model:
                allow_mouse_clicks = not self.game_over and self.game_state.current_player == self.human_player
            else:
                allow_mouse_clicks = not self.auto_play and not self.game_over

            if event.type == pygame.MOUSEBUTTONDOWN and allow_mouse_clicks:
                if event.button == 1:
                    mouse_pos = event.pos
                    legal_actions = self.get_legal_actions()

                    for button in self.buttons:
                        if button['rect'].collidepoint(mouse_pos):
                            action_name = button['action']
                            if any(a.name == action_name for a in legal_actions):
                                action = next(a for a in legal_actions if a.name == action_name)
                                self.step_game(action)
        
        return True
    
    def run(self):
        self.reset_game()
        running = True
        
        while running:
            if self.human_vs_model:
                # In human vs model mode, model plays automatically when it's its turn
                self.model_step()
            elif self.auto_play:
                # In regular mode, auto-play for both players
                self.model_step()

            running = self.handle_events()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--width', type=int, default=1400, help='Window width')
    parser.add_argument('--height', type=int, default=900, help='Window height')
    parser.add_argument('--auto', action='store_true', help='Start in auto-play mode')
    parser.add_argument('--play-delay', type=int, default=1500, help='Delay between actions in auto-play mode (ms)')
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG
    
    agent = PokerTransformerAgent(config)
    
    if args.model:
        if os.path.exists(args.model):
            try:
                checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
                agent.load_state_dict(checkpoint['agent_state'])
                print(f"✓ Loaded model from {args.model}")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                print("Using randomly initialized model instead")
        else:
            print(f"✗ Model file not found: {args.model}")
            print("Using randomly initialized model instead")
    else:
        print("Using randomly initialized model")
    
    visualizer = PygamePokerVisualizer(agent, config, args.width, args.height)
    
    if args.auto:
        visualizer.auto_play = True
    
    visualizer.play_delay = args.play_delay
    visualizer.run()


if __name__ == '__main__':
    main()
