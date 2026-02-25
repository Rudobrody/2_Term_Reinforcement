# %%
import comet_ml
from comet_ml import Experiment 
from torch import nn

import torch
import math
import numpy as np
import pygame
from abstract_car import AbstractCar
from game import Game, PlayerCar2, PlayerCar
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from utils import scale_image
from tqdm import tqdm
import sys
import os
import glob
from collections import deque


# %%
torch.cuda.init()  # Force CUDA initialization

# Verify immediately
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
if device.type == "cuda":
    print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")

GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.35)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.35)
PURPLE_CAR = scale_image(pygame.image.load("imgs/purple-car.png"), 0.35)
GRAY_CAR = scale_image(pygame.image.load("imgs/grey-car.png"), 0.35)


WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

pygame.font.init()  # Initialize the font module
FONT = pygame.font.Font(None, 24)  # Use a default font with size 24


FPS = 60

START_POSITIONS = [
        (180, 200),  
        (150, 200),  
        (180, 160),  
        (150, 160),  
    ]

 # =============================================
# HYPERPARAMETERS — single source of truth
# =============================================
HYPERPARAMS = {
    "episodes": 5000, # 10 000
    "input_dim": 22,
    "action_dim": 5, 
    "epsilon_start": 0.15, # 1
    "epsilon_min": 0.01, # 0.01 
    "epsilon_decay": 0.999, # 0.9995
    "gamma": 0.99,
    "learning_rate": 0.00005, # 0.0003
    "batch_size": 256, # 64
    "buffer_size": 100000,
    "target_sync_freq": 50,
    "save_freq": 500,
    "noise_std": 0.002,
    "architecture": "Dueling Double DQN",
    "optimizer": "Adam",
    "loss": "HuberLoss",
    "random_start": True, # False
    "start_episode": 11000,
    "backtracking": False, # True
    "load_existing_weights": True,
    "weights_to_load_path": 'saved_weights44/grand_champion_model.pth',
    "save_weights_path": "saved_weights45"
}

def create_experiment(project_name="racing-dqn", experiment_name=None):
    """
    Creates and configures a Comet ML experiment.
    
    Args:
        project_name (str): Name of the Comet project
        experiment_name (str): Optional name for this specific run
    
    Returns:
        Experiment object
    """
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=project_name,
        workspace=os.getenv("COMET_WORKSPACE", None), 
        auto_metric_logging=False, 
        auto_param_logging=False,
    )

    if experiment_name:
        experiment.set_name(experiment_name)

    return experiment


def log_hyperparameters(experiment, hyperparams: dict):
    """Logs all hyperparameters at the start of training."""
    experiment.log_parameters(hyperparams)


def log_episode(experiment, metrics: dict, step: int):
    """Logs metrics for a single episode."""
    for key, value in metrics.items():
        experiment.log_metric(key, value, step=step)


def log_model(experiment, model, model_name="racing_dqn"):
    """Logs model architecture as text."""
    experiment.log_text(str(model), metadata={"type": "model_architecture"})


def end_experiment(experiment):
    """Cleanly ends the experiment."""
    experiment.end()

def run_tournament(folders_list, episodes_per_model=50):
    print(f"\nTournament of models...")
    print(f"Every model will be test for {episodes_per_model} epizodes.")
    print(f"Searching for weights in: {folders_list}\n")

    # Search for all files .pth 
    model_files = []
    for folder in folders_list:
        # Searching of files .pth in folder recursively or flatten
        files = glob.glob(f"{folder}/*.pth")
        model_files.extend(files)

    if not model_files:
        print("There is no any .pth file")
        return

    # Sort of files
    model_files.sort()
    
    results = []

    # Initialization of env, once
    game = Game(WIDTH, HEIGHT)
    player_car = PlayerCar2("Tournament_Racer")
    game.add_car(player_car)

    env = RacingEnvWrapper(game, player_car, CHECKPOINTS)
    
    agent = RacingAgent(input_dim=22, action_dim=5)

    # Loop over models
    for weight_path in model_files:
        model_name = os.path.basename(weight_path)
        print(f"Testing: {model_name} ...", end="", flush=True)

        try:
            # Loading weights
            agent.policy_net.load_state_dict(torch.load(weight_path, map_location=agent.device))
            
            # Setting eval mode
            agent.policy_net.eval() 
            
            # Turn of random actions
            agent.epsilon = 0.0 
            
            wins = 0
            total_rewards = []
            steps_list = []
            checkpoints_list = []

            # Loop of races for current models 
            for i in range(episodes_per_model):
                # Reset of env, start from scratch
                state = env.reset(start_checkpoint=0) 

                # Turn of noise
                env.add_noise = False 
                
                done = False
                episode_reward = 0
                steps = 0
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    
                    episode_reward += reward
                    steps += 1
                    state = next_state
                    
                    # Limit of step to avoid inf loop
                    if steps > 2000:
                        done = True

                if info.get('finished', False):
                    wins += 1
                
                total_rewards.append(episode_reward)
                steps_list.append(steps)
                checkpoints_list.append(info.get('checkpoint_reached', 0) + env.car.checkpoint_index)

            # Counting metrics
            avg_reward = sum(total_rewards) / len(total_rewards)
            avg_steps = sum(steps_list) / len(steps_list)
            win_rate = (wins / episodes_per_model) * 100
            
            print(f"Result: {win_rate:.1f}% Win Rate | Avg Reward: {avg_reward:.1f}")
            
            results.append({
                "Model": model_name,
                "Path": weight_path,
                "Win Rate (%)": win_rate,
                "Avg Reward": avg_reward,
                "Avg Steps (Speed)": avg_steps,
                "Avg Checkpoints": sum(checkpoints_list) / len(checkpoints_list)
            })

        except Exception as e:
            print(f"Error of loading: {e}")

    # Summary
    print("\n" + "="*60)
    print("END RESULTS OF TOURNAMENT")
    print("="*60)

    # Sorting over win_rate and then average reward
    results.sort(key=lambda x: (x["Win Rate (%)"], x["Avg Reward"]), reverse=True)

    print(f"{'Model':<35} | {'Win %':<8} | {'Reward':<8} | {'Steps':<8}")
    print("-" * 65)
    for res in results:
        print(f"{res['Model']:<35} | {res['Win Rate (%)']:<6.1f}% | {res['Avg Reward']:<8.1f} | {res['Avg Steps (Speed)']:<8.1f}")
    
    print("="*60)
    best_model = results[0]
    print(f"The best results got a model: {best_model['Model']}")
    print(f"Path to best model: {best_model['Path']}")
    print("="*60)

class MetricsTracker:
    """
    Tracks and computes running statistics for training.
    Feeds data to Comet ML.
    """
    def __init__(self, window_size=100):
        self.window_size = window_size

        # Per-episode raw data
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_checkpoints = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_wall_hits = deque(maxlen=window_size)
        self.episode_car_collisions = deque(maxlen=window_size)
        self.episode_max_velocities = deque(maxlen=window_size)
        self.episode_finished = deque(maxlen=window_size)  # Did the car complete the track?

        # Training internals
        self.losses = deque(maxlen=window_size)

        # Current episode accumulators
        self.current_reward = 0
        self.current_steps = 0
        self.current_checkpoints = 0
        self.current_wall_hits = 0
        self.current_car_collisions = 0
        self.current_max_velocity = 0
        self.current_finished = False

    def reset_episode(self):
        """Call at the start of each episode."""
        self.current_reward = 0
        self.current_steps = 0
        self.current_checkpoints = 0
        self.current_wall_hits = 0
        self.current_car_collisions = 0
        self.current_max_velocity = 0
        self.current_finished = False

    def step(self, reward, velocity=0, wall_hit=False, car_collision=False, checkpoint_reached=False):
        """Call after every environment step."""
        self.current_reward += reward
        self.current_steps += 1
        self.current_max_velocity = max(self.current_max_velocity, velocity)

        if wall_hit:
            self.current_wall_hits += 1
        if car_collision:
            self.current_car_collisions += 1
        if checkpoint_reached:
            self.current_checkpoints += 1

    def end_episode(self, finished=False):
        """Call at the end of each episode. Stores results."""
        self.current_finished = finished
        self.episode_rewards.append(self.current_reward)
        self.episode_checkpoints.append(self.current_checkpoints)
        self.episode_lengths.append(self.current_steps)
        self.episode_wall_hits.append(self.current_wall_hits)
        self.episode_car_collisions.append(self.current_car_collisions)
        self.episode_max_velocities.append(self.current_max_velocity)
        self.episode_finished.append(1 if finished else 0)

    def log_loss(self, loss_value):
        """Call after each training step."""
        self.losses.append(loss_value)

    def get_episode_metrics(self, epsilon, episode):
        """
        Returns a dictionary of all metrics for Comet ML logging.
        """
        metrics = {
            # Current episode
            "episode/reward": self.current_reward,
            "episode/checkpoints": self.current_checkpoints,
            "episode/length": self.current_steps,
            "episode/wall_hits": self.current_wall_hits,
            "episode/car_collisions": self.current_car_collisions,
            "episode/max_velocity": self.current_max_velocity,
            "episode/finished": 1 if self.current_finished else 0,

            # Rolling averages
            "rolling/reward_avg": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "rolling/reward_max": np.max(self.episode_rewards) if self.episode_rewards else 0,
            "rolling/checkpoints_avg": np.mean(self.episode_checkpoints) if self.episode_checkpoints else 0,
            "rolling/checkpoints_max": np.max(self.episode_checkpoints) if self.episode_checkpoints else 0,
            "rolling/length_avg": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "rolling/wall_hits_avg": np.mean(self.episode_wall_hits) if self.episode_wall_hits else 0,
            "rolling/car_collisions_avg": np.mean(self.episode_car_collisions) if self.episode_car_collisions else 0,
            "rolling/finish_rate": np.mean(self.episode_finished) if self.episode_finished else 0,

            # Training
            "training/epsilon": epsilon,
            "training/loss_avg": np.mean(self.losses) if self.losses else 0,
            "training/episode": episode,
        }

        return metrics

    def get_best_reward(self):
        return max(self.episode_rewards) if self.episode_rewards else float('-inf')
    

track_path =  [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]



# Interpolate evenly spaced checkpoints
def generate_checkpoints(track_path, num_checkpoints=100):
    checkpoints = []
    for i in range(len(track_path) - 1):
        x1, y1 = track_path[i]
        x2, y2 = track_path[i + 1]
        for t in np.linspace(0, 1, num_checkpoints // len(track_path)):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            checkpoints.append((int(x), int(y)))
    return checkpoints


CHECKPOINTS = generate_checkpoints(track_path)

def draw_checkpoints(win, checkpoints):
    for x, y in checkpoints:
        pygame.draw.circle(win, (0, 255, 0), (x, y), 5)

class RacingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RacingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value Stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage Stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

class RacingAgent:
    def __init__(self, input_dim=22, action_dim=5):
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        self.epsilon = HYPERPARAMS['epsilon_start']
        self.epsilon_min = HYPERPARAMS['epsilon_min']
        self.epsilon_decay = HYPERPARAMS['epsilon_decay']
        self.gamma = HYPERPARAMS["gamma"]
        self.memory = deque(maxlen=HYPERPARAMS['buffer_size'])
        self.batch_size = HYPERPARAMS['batch_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = RacingDQN(input_dim, action_dim).to(self.device)
        self.target_net = RacingDQN(input_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=HYPERPARAMS['learning_rate'])

        # HubberLoss more stable than MSE for RL
        self.loss_fn = nn.HuberLoss() 
        

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def cache(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        # Warmup Check
        # Wait until we have enough data (e.g., 2,000 samples)
        if len(self.memory) < 2000:
            return

        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.stack(state).to(self.device)
        action = torch.tensor(action).unsqueeze(1).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        # Double DQN Logic
        # Select best action using Policy Net
        with torch.no_grad():
            next_actions = self.policy_net(next_state).argmax(1).unsqueeze(1)
            
            # Evaluate that action using Target Net
            next_q_values = self.target_net(next_state).gather(1, next_actions).squeeze(1)
            target_q = reward + (1 - done) * self.gamma * next_q_values

        current_q = self.policy_net(state).gather(1, action).squeeze(1)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
    

    def sync(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    
class RacingEnvWrapper:
    """
    Wraps the Pygame Game logic to provide a step-by-step RL interface.
    """
    def __init__(self, game_instance, agent_car, checkpoints, is_test_mode=False):
        self.is_test_mode = is_test_mode
        self.game = game_instance
        self.car = agent_car
        self.checkpoints = checkpoints

        self.ACTIONS = ["forward", "backward", "left", "right", "stop"]
        self.action_space_size = 5 # Forward, Backward, Left, Right, Stop
        self.observation_space_size = 22 # 8 walls + 8 cars + 4 kinematic info

        # Internal state for reward calculation
        self.last_checkpoint_idx = 0
        self.last_position = (0, 0)
        self.wall_hits = 0
        self.steps_without_progress = 0
        self.last_distance_to_checkpoint = None

        # Flat to detect hitting wall
        self.is_touching_wall = False


    def _check_car_collisions(self):
        """Check collision against all other cars in the game."""
        for other_car in self.game.cars:
            if other_car is not self.car:  # Don't check self
                if self.car.collide_car(other_car):
                    return True
                
        return False
    

    def _calculate_min_ttc(self, wall_dists):
        # If car is just standing then return Time to Crush inf
        if self.car.vel <= 0.1:
            return float('inf')

    
        ray_angles_relative = [0, 45, 90, 135, 180, 225, 270, 315]
        
        min_ttc = float('inf')
        
        # Vector of velocity of the car
        car_rad = math.radians(self.car.angle + 90)
        vel_x = math.cos(car_rad) * self.car.vel
        vel_y = -math.sin(car_rad) * self.car.vel

        for i, dist in enumerate(wall_dists):
            # Absolute angle 
            ray_angle_abs = math.radians(self.car.angle + 90 - ray_angles_relative[i])
            
            # Vector of radius - we use radius to calculate distance to wall
            ray_vec_x = math.cos(ray_angle_abs)
            ray_vec_y = -math.sin(ray_angle_abs)
            
            # Velocity mapped on our radius 
            v_proj = (vel_x * ray_vec_x) + (vel_y * ray_vec_y)
            
            # if we are getting closer
            if v_proj > 0.01:
                # We normalized our distance / 1000 so now reverse 
                real_dist = dist * 1000.0 
                ttc = real_dist / v_proj
                
                if ttc < min_ttc:
                    min_ttc = ttc

        return min_ttc
    

    def _car_center(self):
        """Returns the center position of the car sprite."""
        cx = self.car.x + self.car.img.get_width() // 2
        cy = self.car.y + self.car.img.get_height() // 2
        return cx, cy

    def _distance_to_next_checkpoint(self):
        if self.car.checkpoint_index < len(self.checkpoints):
            cp = self.checkpoints[self.car.checkpoint_index]
            return math.sqrt((self.car.x - cp[0])**2 + (self.car.y - cp[1])**2)
        return 0

    def reset(self, start_checkpoint=None):
        self.car.reset()
        self.wall_hits = 0
        self.steps_without_progress = 0
        self.is_touching_wall = False
        self.last_distance_to_checkpoint = None

        BAD_CHECKPOINTS = [61, 63, 65, 77]

        def _place_car_at(idx):
            """Helper: places car at given checkpoint index with proper angle."""
            if idx in BAD_CHECKPOINTS:
                idx = max(idx - 1, 0)

            if idx == 0:
                start_pos = random.choice(START_POSITIONS)
            else:
                start_pos = self.checkpoints[idx]

            self.car.x = start_pos[0]
            self.car.y = start_pos[1]
            self.car.checkpoint_index = idx

            # Point car towards next checkpoint
            next_idx = min(idx + 1, len(self.checkpoints) - 1)
            next_cp = self.checkpoints[next_idx]
            dx = next_cp[0] - start_pos[0]
            dy = next_cp[1] - start_pos[1]
            self.car.angle = math.degrees(math.atan2(-dx, -dy))
            self.car.vel = 0

            self.last_checkpoint_idx = idx
            self.last_distance_to_checkpoint = self._distance_to_next_checkpoint()

        def _apply_random_scenario(idx):
            """Helper: randomizes angle/velocity for training variety."""
            scenario_roll = random.random()

            if scenario_roll < 0.8:
                # Scenario A (80%): normal drive, slight noise
                next_idx = min(idx + 1, len(self.checkpoints) - 1)
                next_cp = self.checkpoints[next_idx]
                dx = next_cp[0] - self.car.x
                dy = next_cp[1] - self.car.y
                ideal_angle = math.degrees(math.atan2(-dx, -dy))
                self.car.angle = ideal_angle + random.uniform(-10, 10)
                self.car.vel = random.uniform(0, 2.0)
            else:
                # Scenario B (20%): recovery training (bad angle, negative vel)
                self.car.angle = random.uniform(0, 360)
                self.car.vel = -random.uniform(1.0, 5.0)

       
        # If start_checkpoint is explicitly given, start from the checkpoint
        if start_checkpoint is not None:
            _place_car_at(start_checkpoint)
            return self._get_state()

        
        # Fixed start (no random, no backtracking)
        if not HYPERPARAMS['random_start'] and not HYPERPARAMS['backtracking']:
            _place_car_at(0)
            return self._get_state()

        # Backtracking
        elif not HYPERPARAMS['random_start'] and HYPERPARAMS['backtracking']:
            idx = 0
            _place_car_at(idx)
            _apply_random_scenario(idx)
            self.last_distance_to_checkpoint = self._distance_to_next_checkpoint()
            return self._get_state()

        # Random start
        elif HYPERPARAMS['random_start']:
            chosen_idx = random.randint(0, len(self.checkpoints) - 1)
            _place_car_at(chosen_idx)
            _apply_random_scenario(chosen_idx)
            self.last_distance_to_checkpoint = self._distance_to_next_checkpoint()
            return self._get_state()

        # Fallback safety net
        _place_car_at(0)
        return self._get_state()

    def step(self, action_idx):
        # Making action 
        action_str = self.ACTIONS[action_idx] 
        self.car.update_progress(self.checkpoints)
        self.car.perform_action(action_str)
        
        # Check if we are touching wall right now
        is_currently_touching = self.car.collide(TRACK_BORDER_MASK)

        # Check collision before rewards
        self.game.check_collisions()
        
        finish_poi = self.car.collide(FINISH_MASK, *FINISH_POSITION)
        if finish_poi is not None:
            if finish_poi[1] == 0:
                self.car.bounce()

        # Initialization
        reward = 0
        done = False
        info = {}
        checkpoint_reached = False
        wall_hit_flag = False # Temp flag for info dict
        finished = False


        if is_currently_touching:
            if not self.is_touching_wall:
                # Penalty for hitting wall from higgh speed
                reward -= 5 + abs(self.car.vel) * 2
                
                self.wall_hits += 1
                wall_hit_flag = True # Mark for info
                
                # Check for game over (too many crashes)
                if self.wall_hits > 10:
                    done = True
            else:
                # Penalty for still touching
                reward -= 0.3 

        # Update rising edge memory
        self.is_touching_wall = is_currently_touching

        # Vector to target
        # In state[2] we got distance/index so we have to calculate to vector
        target_idx = min(int(self.car.checkpoint_index), len(self.checkpoints) - 1)
        target_pos = self.checkpoints[target_idx]
        
        dx = target_pos[0] - self.car.x
        dy = target_pos[1] - self.car.y

        # Normalization
        dist_to_cp = math.sqrt(dx**2 + dy**2)
        if dist_to_cp == 0: dist_to_cp = 1
        target_vec_x = dx / dist_to_cp
        target_vec_y = dy / dist_to_cp
        
        # Projected velocity
        car_rad = math.radians(self.car.angle + 90)
        vel_vec_x = math.cos(car_rad) * self.car.vel
        vel_vec_y = -math.sin(car_rad) * self.car.vel
        
        # A velocity in the direction of target
        proj_velocity = (vel_vec_x * target_vec_x) + (vel_vec_y * target_vec_y)
        
        # Reward for velocity
        reward += (proj_velocity / self.car.max_vel) * 0.5 
        
        # Analize wall rays thanks to state[0]
        wall_dists = self.car.get_rays_and_distances(TRACK_BORDER_MASK)[1] 

        MIN_SAFE_DIST = 18.0 
        closest_wall = min(wall_dists) if wall_dists else 1000.0
        
        if closest_wall < MIN_SAFE_DIST:
            # Kara rośnie im mocniej auto "wciska" się w bąbel
            bubble_risk = (MIN_SAFE_DIST - closest_wall) / MIN_SAFE_DIST
            reward -= bubble_risk * 0.5 

        # Time to collision
        ttc = self._calculate_min_ttc(wall_dists)
        DANGER_ZONE = 7.0 
        
        if ttc < DANGER_ZONE and self.car.vel > 1.0:
            risk_factor = (DANGER_ZONE - ttc) / DANGER_ZONE
            smooth_penalty = (risk_factor ** 2) * 0.3
            reward -= smooth_penalty
        
        # Analize distance to other cars thanks to state[1] 
        car_dists = self.car.get_distances_to_cars(self.game.cars)
        
        # Filtering
        visible_cars = [d for d in car_dists if d < 190] 
        
        # Collision with other car
        if self._check_car_collisions():

            # Reward is less than hitting a wall
            reward -= 3.0 
        
        # Reward for progress
        current_dist = self._distance_to_next_checkpoint()
        if self.last_distance_to_checkpoint is not None:
            progress = self.last_distance_to_checkpoint - current_dist
            reward += progress * 0.15

        self.last_distance_to_checkpoint = current_dist

        # Reward for reaching checkpoints
        if self.car.checkpoint_index > self.last_checkpoint_idx:
            reward += 50.0
            self.last_checkpoint_idx = self.car.checkpoint_index
            self.steps_without_progress = 0
            checkpoint_reached = True
            
        # Reward for finishing the race
        if self.car.checkpoint_index >= len(self.checkpoints) - 1:
            # Bonus for fast finish
            speed_bonus = max(0, 500 - self.steps_without_progress)
            reward += 500.0 + speed_bonus  
            done = True
            finished = True
            

        # Penalty for no proggress
        if self.steps_without_progress >= 50:
        
            # This will be our factor
            steps_over_limit = self.steps_without_progress - 50

            normalized_time = steps_over_limit / 60 
            raw_penalty = 0.05 * (normalized_time**2)
            clipped_penalty = min(raw_penalty, 2)
            reward -= clipped_penalty

        
        # Negative reward for timeout
        if self.steps_without_progress > 500: # if 60 fps so >8sec without reward
            reward -= 5.0
            if not self.is_test_mode:
                done = True
        
        diff = abs(wall_dists[2] - wall_dists[6])
        centering_factor = max(0, 1-diff)
        reward += 0.05*centering_factor
        
        # Time penalty
        self.steps_without_progress += 1
        reward -= 0.05 # Time penalty

        info = {
            "checkpoint_reached": checkpoint_reached,
            "wall_hit": wall_hit_flag,
            "car_collision": None, #car_collision,
            "finished": finished,
            "velocity": self.car.vel,
        }
        
        return self._get_state(), reward, done, info

    def set_start_checkpoint(self, checkpoint_idx):
        self.fixed_start_checkpoint = checkpoint_idx


    def _get_state(self, add_noise=False):
        """
        Constructs the state vector for the Neural Network.
        Size: 22
        """
        # Raycasts to Walls (Normalized 0-1)
        _, wall_dists = self.car.get_rays_and_distances(TRACK_BORDER_MASK)
        

        # If rays are missing (glitch), pad them with max distance (1000)
        expected_rays = 8
        if len(wall_dists) < expected_rays:
            wall_dists = wall_dists + [1000] * (expected_rays - len(wall_dists))
        elif len(wall_dists) > expected_rays:
             wall_dists = wall_dists[:expected_rays]
        
        wall_dists = np.array(wall_dists) / 1000.0 # Max ray length

        # Raycasts to other cars (Normalized)
        car_dists = self.car.get_distances_to_cars(self.game.cars)

        if len(car_dists) < expected_rays:
             car_dists = car_dists + [200] * (expected_rays - len(car_dists))
        elif len(car_dists) > expected_rays:
             car_dists = car_dists[:expected_rays]

        car_dists = np.array(car_dists) / 200.0 # Max car scan dist

        # Kinematics
        velocity = self.car.vel / self.car.max_vel
        
        angle_to_target = 0.0
        dist_to_target = 0.0

        # Target orientation (angle to next checkpoint)
        # because the car needs to know where to go.
        if self.car.checkpoint_index < len(self.checkpoints):
            target = self.checkpoints[self.car.checkpoint_index]

            
            dx = target[0] - self.car.x
            dy = target[1] - self.car.y

            target_angle = math.degrees(math.atan2(-dx, -dy))

            # Relative angle
            diff_angle = (target_angle - self.car.angle) % 360
            if diff_angle > 180: diff_angle -= 360
            angle_to_target = diff_angle / 180.0 # Normalize -1 to 1
            
            dist_to_target = math.sqrt(dx**2 + dy**2) / 1000.0
        else:
            angle_to_target = 0

        checkpoint_progress = self.car.checkpoint_index / len(self.checkpoints)
        car_angle = (self.car.angle % 360) / 360.0

        state = np.concatenate([
        wall_dists,              # 8
        car_dists,               # 8
        [velocity],              # 1
        [angle_to_target],       # 1
        [dist_to_target],        # 1
        [checkpoint_progress],   # 1
        [car_angle],             # 1
        [1.0 if self.car.vel >= 0 else 0.0],  # 1
    ])  # Total: 22


        if len(state) != 22:
            print(f"CRITICAL: State size is {len(state)}! Padding with zeros.")
            # Create a zero vector of the correct size to prevent crash
            state = np.zeros(22)


        if getattr(self, 'add_noise', True):  # Reads the attribute you set
            noise = np.random.normal(0, 0.002, size=state.shape)
            state = state + noise
        
        return torch.tensor(state, dtype=torch.float32)
    
def evaluate_agent(env, agent, n_eval_episodes=5):
    """
    Launch agent in test mode without noise and epsilon=0 always from the beggining of the track
    It returns average checkpoints score and percent of finishes
    """
    # Store old epislon, because we wanna return to it after refill of buffer
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0 # Apply only knowledge
    
    success_count = 0
    total_checkpoints_reached = 0
    
    for _ in range(n_eval_episodes):
        # Start from the beginning
        state = env.reset(start_checkpoint=0)
        done = False
        
        env_noise_backup = getattr(env, 'add_noise', False)
        env.add_noise = False 
        
        while not done:
            action_idx = agent.act(state)
            state, _, done, info = env.step(action_idx)
            
            # Secure infinity loop
            if env.steps_without_progress > 500:
                done = True
        
        # To stastistics
        if info.get('finished', False):
            success_count += 1
        
        total_checkpoints_reached += env.car.checkpoint_index
        
        # Add noise
        env.add_noise = env_noise_backup

    # Restore epsilon
    agent.epsilon = old_epsilon
    
    avg_progress = total_checkpoints_reached / n_eval_episodes
    success_rate = (success_count / n_eval_episodes) * 100
    
    return avg_progress, success_rate



def save_debug_snapshot(game, car, checkpoints, episode_num, folder="debug_snapshots"):
    """
    Checking respawn
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    surface = pygame.Surface((WIDTH, HEIGHT))
    
    # Draw track
    surface.blit(TRACK, (0, 0))
    
    # Drawc ar
    car.draw(surface)
    
    car_cx = car.x + car.img.get_width() / 2
    car_cy = car.y + car.img.get_height() / 2
    
    # Target
    target_idx = car.checkpoint_index
    if target_idx >= len(checkpoints): target_idx = 0
    target_pos = checkpoints[target_idx]

    # Target line
    pygame.draw.line(surface, (0, 255, 0), (car_cx, car_cy), target_pos, 3)
    pygame.draw.circle(surface, (0, 255, 0), target_pos, 8) 

    # Where is car looking
    angle_rad = math.radians(car.angle + 90) # +90 bo w Pygame 0 to góra
    line_len = 80
    
    end_x = car_cx + math.cos(angle_rad) * line_len
    end_y = car_cy - math.sin(angle_rad) * line_len
    
    pygame.draw.line(surface, (255, 0, 0), (car_cx, car_cy), (end_x, end_y), 3)

    # Add fonts
    font = pygame.font.SysFont('Arial', 20)
    text = font.render(f"Epizod: {episode_num} | CP: {target_idx}", True, (255, 255, 255))
    surface.blit(text, (10, 10))

    # Save file
    filename = f"{folder}/respawn_ep_{episode_num}.png"
    pygame.image.save(surface, filename)
    
def train_racing_car():
    
    
    # COMET ML SETUP
    experiment = create_experiment(
        project_name="racing-dqn",
        experiment_name="w45_Backtracking z TTC penalty, fix bez centrowania, dodane pozycje startowe, check finish line, rebalance nagrod, pre fill buffer"
    )
    log_hyperparameters(experiment, HYPERPARAMS)

    
    # GAME & AGENT SETUP
    game = Game(WIDTH, HEIGHT)
    player_car = PlayerCar2("AI_Agent")
    game.add_car(player_car)

    env = RacingEnvWrapper(game, player_car, CHECKPOINTS)
    agent = RacingAgent(
        input_dim=HYPERPARAMS["input_dim"],
        action_dim=HYPERPARAMS["action_dim"]
    )
    
    
    if HYPERPARAMS['load_existing_weights']:
        try:
            # Load weights into Policy Net
            agent.policy_net.load_state_dict(torch.load(HYPERPARAMS['weights_to_load_path'], map_location=agent.device))
            
            # Sync Target Net immediately
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            old_epsilon = agent.epsilon
            agent.epsilon = 0.1  # Mało losowy, zbieramy "dobre" doświadczenia

            for _ in tqdm(range(50), desc="Pre-filling buffer"):
                state = env.reset()
                done = False
                steps = 0
                while not done and steps < 600:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.cache(state, action, reward, next_state, done)
                    state = next_state
                    steps += 1

            agent.epsilon = old_epsilon
            print(f"Buffer filled with {len(agent.memory)} transitions")

            # Update of the start episode for comet ML
            HYPERPARAMS['start_episode'] = 9000
            print(f"Continuing training from Episode {HYPERPARAMS['start_episode']}")

            print(f"Successfully loaded weights from {HYPERPARAMS['weights_to_load_path']}")
            print(f"Reduced Epsilon to {agent.epsilon:.2f} to continue training.")
            
        except FileNotFoundError:
            print(f"Weights file not found at {HYPERPARAMS['weights_to_load_path']}. Starting fresh.")

    # Log model architecture
    log_model(experiment, agent.policy_net)

    # Metrics tracker
    tracker = MetricsTracker(window_size=100)

    # Best model tracking
    best_reward = float('-inf')

    total_checkpoints = len(env.checkpoints)
    current_start_idx = max(total_checkpoints - 5, 0)
    

    # Dict to keep best results for every level if we using backtracking
    best_rewards_per_level = {}
    best_rewards_per_level[current_start_idx] = float('-inf')

    # Counter of finishes in a row
    consecutive_successes = 0

    # Counter
    epoch_to_learn_from_start = 0

    # Initialization of reward for phase of eval
    best_eval_score = float('-inf')

    
    # TRAINING LOOP
    for episode in range(HYPERPARAMS['start_episode'], HYPERPARAMS['start_episode']+HYPERPARAMS["episodes"]):
        
        # Reset of env
        if HYPERPARAMS['backtracking']:
            if current_start_idx == 0:
                current_start_idx = random.choice(START_POSITIONS)
            else:
                state = env.reset(start_checkpoint=current_start_idx)
        else:    
            state = env.reset(start_checkpoint=None) 
        
        if episode < 20 or episode % 100 == 0:
            save_debug_snapshot(game, env.car, env.checkpoints, episode)
            if episode < 20:
                print(f"Zapisano podgląd startu: debug_snapshots/respawn_ep_{episode}.png")

        # Reset of some metrics 
        tracker.reset_episode()
        done = False

        while not done:
            # Handle Pygame events
            for event in pygame.event.get([pygame.QUIT, pygame.KEYDOWN]):
                if event.type == pygame.QUIT:
                    # Save before quitting
                    torch.save(agent.policy_net.state_dict(), f"{HYPERPARAMS['save_weights_path']}/emergency_save.pth")
                    end_experiment(experiment)
                    pygame.quit()
                    return

            # take an action
            action_idx = agent.act(state)
            
            # Environment step
            next_state, reward, done, info = env.step(action_idx)

            # Store and learn
            agent.cache(state, action_idx, reward, next_state, done)
            loss = agent.learn()

            # Track metrics
            tracker.step(
                reward=reward,
                velocity=info.get("velocity", 0),
                wall_hit=info.get("wall_hit", False),
                car_collision=info.get("car_collision", False),
                checkpoint_reached=info.get("checkpoint_reached", False),
            )
            if loss is not None:
                tracker.log_loss(loss)

            state = next_state

        
        tracker.end_episode(finished=info.get("finished", False))
        
        if info.get('finished') or env.car.checkpoint_index >= total_checkpoints - 1:
            consecutive_successes += 1
        
        
        # If there were 3 win in a row
        if consecutive_successes >= 5:
            if current_start_idx > 0:
                
                # We move back with checkpoint
                current_start_idx -= 5 # Cofamy się o piec checkpoint
                current_start_idx = max(current_start_idx, 0)

                if current_start_idx == 0:
                    epoch_to_learn_from_start += 1 

                    # Give him 1000 epochs to learn start from starting positions
                    if epoch_to_learn_from_start == 1000:
                        HYPERPARAMS['backtracking'] = False
                        HYPERPARAMS['random_start'] = True

                # Reset of counter, because we have new level
                consecutive_successes = 0 

                # best reward for new lvl
                if current_start_idx not in best_rewards_per_level:
                    best_rewards_per_level[current_start_idx] = float('-inf')
                
                

        # Decay epsilon ONCE per episode
        agent.decay_epsilon()

        
        # Sync target network periodically
        if episode % HYPERPARAMS["target_sync_freq"] == 0:
            agent.sync()

        # log to comet ml
        metrics = tracker.get_episode_metrics(agent.epsilon, episode)

        # Information about current lvl of backtracking for comet
        metrics["training/start_backtracking_checkpoint_idx"] = current_start_idx
        log_episode(experiment, metrics, step=episode)

        # Console output every 10 episodes
        if episode % 10 == 0:
            print(
                f"Ep {episode:>5d} | "
                f"Reward: {metrics['episode/reward']:>8.1f} | "
                f"Avg: {metrics['rolling/reward_avg']:>8.1f} | "
                f"CPs: {metrics['episode/checkpoints']:>3.0f} | "
                f"Avg CPs: {metrics['rolling/checkpoints_avg']:>5.1f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Finish%: {metrics['rolling/finish_rate']*100:.1f}%"
            )

        if episode % 100 == 0 and episode > 0:
            print(f"ROZPOCZYNAM EWALUACJĘ (Epizod {episode})...")
            
            # Testing agent on full track
            avg_cp, win_rate = evaluate_agent(env, agent, n_eval_episodes=5)
            
            print(f"Wynik testu: Avg Checkpoint: {avg_cp:.1f} | Win Rate: {win_rate:.0f}%")
            
            # Logs to comet
            experiment.log_metric("eval/avg_checkpoint_from_start", avg_cp, step=episode)
            experiment.log_metric("eval/win_rate_from_start", win_rate, step=episode)
            
            
            current_eval_score = win_rate * 1000 + avg_cp 
            
            if current_eval_score > best_eval_score:
                best_eval_score = current_eval_score
                
                
                path = f"{HYPERPARAMS['save_weights_path']}/grand_champion_model.pth"
                torch.save(agent.policy_net.state_dict(), path)
                print(f"NOWY MISTRZ, zapisano: grand_champion_model.pth")
                
                # Jeśli agent potrafi przejechać całą trasę (100% win rate), 
                # to warto go zapisać też osobno, żeby go nie zgubić
                if win_rate == 100.0:
                    torch.save(agent.policy_net.state_dict(), f"{HYPERPARAMS['save_weights_path']}/perfect_run_model.pth")

        

        # Regular checkpoints
        if episode % HYPERPARAMS["save_freq"] == 0:
            path = f"{HYPERPARAMS['save_weights_path']}/racing_agent_ep{episode}.pth"
            torch.save(agent.policy_net.state_dict(), path)
            experiment.log_model(f"checkpoint_ep{episode}", path)
        
        
    # End of training
    final_path = f"{HYPERPARAMS['save_weights_path']}/racing_agent_final.pth"
    torch.save(agent.policy_net.state_dict(), final_path)
    experiment.log_model("final_model", final_path)

    end_experiment(experiment)
    print(f"\nTraining complete! Best avg reward: {best_reward:.1f}")

if __name__ == "__main__":
   
    if len(sys.argv) > 1 and sys.argv[1] == "tournament":
        # We just give a name of folder where are our weights to compare
        folders_to_scan = [
            "saved_weights9", "saved_weights10", "saved_weights11", "saved_weights12_bouncing", "saved_weights13_bouncing", "saved_weights20",
            "saved_weights21", "saved_weights22", "saved_weights30", "saved_weights31", "saved_weights32", "saved_weights40",
            "saved_weights41", "saved_weights42", "saved_weights43",
            
            ] 
        
        # Run torunament, 20 episodes per one model
        run_tournament(folders_to_scan, episodes_per_model=20)
    
    else:
        # Default: train
        train_racing_car()


# %%
