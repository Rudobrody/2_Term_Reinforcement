import torch 
import torch.nn as nn
import os 
#from game import AbstractCar, CHECKPOINTS
from abstract_car import AbstractCar
import numpy as np
import math


#WEIGHTS_FILE_v1 = r'grand_champion_model_v1.pth' # Z tymi wagami zdobyl 7 punktów xDDD
#WEIGHTS_FILE_v1 = r'grand_champion_model_41_9000.pth' # zle sobie radzi ze scianami i ze startem
#WEIGHTS_FILE_v1 = r'D:\NewWorkspaceVSC\Reinforcement\2_Term_Reinforcement\Cars\saved_weights42\grand_champion_model.pth' # mozna dotrenowac
WEIGHTS_FILE_v1= r'D:\NewWorkspaceVSC\Reinforcement\2_Term_Reinforcement\Cars\saved_weights45\racing_agent_ep11000.pth'


class PlayerCar2(AbstractCar):
    def __init__(self, name):
        super().__init__(name)
        
        # cpu for safety during tournament
        self.device = torch.device("cpu") 
        
        # Initialization
        self.model = RacingDQN(input_dim=22, output_dim=5).to(self.device)

        # Loading weights 
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(script_dir, WEIGHTS_FILE_v1)
            
            if os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                self.model.eval()
                print(f"[{name}] Wagi zaladowane: {WEIGHTS_FILE_v1}")
            else:
                print(f"[{name}] Brak pliku wag: {WEIGHTS_FILE_v1}")
        except Exception as e:
            print(f"[{name}] Blad ładowania wag: {e}")

        # Defining actions
        self.ACTIONS = ["forward", "backward", "left", "right", "stop"]

        # Variable in case of being stucked
        self.stuck_counter = 0
        self.reverse_timer = 0
        
        # Variable for retargetting
        self.last_target_idx = -1
        self.steps_no_progress = 0
        
        self.retarget_timer = 0
        self.emergency_target_idx = 0

    def choose_action(self, state):
        """
        Determines the next action for the car based on the current state of the environment.

        Parameters:
            state (list): A 3-element list representing the car's current state:
                - state[0]: A list of 8 float values representing distances to the track border
                            in 8 directions (every 45 degrees, starting from forward).
                - state[1]: A list of 8 float values representing distances to the nearest car
                           in the same 8 directions.
                - state[2]: A 2-element list representing progress information:
                            - state[2][0]: The index of the closest checkpoint.
                            - state[2][1]: The car's progress, e.g., distance to the next checkpoint
                                           or normalized progress value.

        Returns:
            - "forward": Move the car forward.
            - "backward": Move the car backward.
            - "left": Turn the car left.
            - "right": Turn the car right.
            - "stop": Reduce the car's speed.
            
        """
        
        # Check are we during backward action
        if self.reverse_timer > 0:
            self.reverse_timer -= 1
            return "backward"
        
        # If we are stuck
        if abs(self.vel) < 0.2: 
            self.stuck_counter += 1
        else:
            # If car goes normaly way we reset the counter
            self.stuck_counter = 0

        # If car is stuck over half sec, because FPS is 60
        if self.stuck_counter > 30:
            self.stuck_counter = 0
            # Set backward action for 1/3 sec
            self.reverse_timer = 20 
            return "backward"
        
    
        wall_dists = list(state[0])
        if len(wall_dists) < 8:
            wall_dists += [1000.0] * (8 - len(wall_dists))
        elif len(wall_dists) > 8:
            wall_dists = wall_dists[:8]
        wall_dists = np.array(wall_dists) / 1000.0

        car_dists = np.ones(8, dtype=np.float32)

        
        # Physics
        velocity = self.vel / self.max_vel
        
        # Calculateing the angle and distance to the next target
        target_idx = int(state[2][0])
        
        angle_to_target = 0.0
        dist_to_target = 0.0

        # Retarget
        car_cx = self.x + self.img.get_width() // 2
        car_cy = self.y + self.img.get_height() // 2
        
        checkpoints_list = state[3]
        
        # Actual target from env
        actual_target_idx = int(state[2][0]) if isinstance(state[2], (list, tuple)) else int(state[2])
        if actual_target_idx == self.last_target_idx:
            # If agent is still trying to catch the same checkpoint
            self.steps_no_progress += 1
        else:
            # Agent catched checkpoints so increment
            self.steps_no_progress = 0
            self.last_target_idx = actual_target_idx
            self.retarget_timer = 0 

        # If after 2 sec we do not reached checkpoints then retarget
        if self.steps_no_progress > 180:
            self.steps_no_progress = 0  # Reset of counter to give him new chance 
            self.retarget_timer = 60    # Active retarggeting for 1 sec
            
            # Looking for closest chechpoint
            start_idx = max(0, actual_target_idx - 5)
            end_idx = min(len(checkpoints_list), actual_target_idx + 2)
            
            min_dist = float('inf')
            closest_idx = actual_target_idx
            
            for i in range(start_idx, end_idx):
                cp = checkpoints_list[i]
                dist = math.sqrt((cp[0] - car_cx)**2 + (cp[1] - car_cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    
            self.emergency_target_idx = closest_idx

        # Seting target for network
        if self.retarget_timer > 0:
            self.retarget_timer -= 1
            target_idx = self.emergency_target_idx
        else:
            target_idx = actual_target_idx

    
        if target_idx < len(checkpoints_list):
            
            # Definine position of the next target
            target_pos = checkpoints_list[target_idx]
            
            dx = target_pos[0] - self.x
            dy = target_pos[1] - self.y
            
            target_angle = math.degrees(math.atan2(-dx, -dy))

            # Relative angle
            diff_angle = (target_angle - self.angle) % 360

            if diff_angle > 180: diff_angle -= 360
            
            # Normalize -1 to 1
            angle_to_target = diff_angle / 180.0
            
            # Distance
            dist_to_target = math.sqrt(dx**2 + dy**2) / 1000.0
        else:
            angle_to_target = 0
            
        checkpoint_progress = target_idx / len(checkpoints_list)
        car_angle = (self.angle % 360) / 360.0
        
        # Build Tensor
        state_vector = np.concatenate([
            wall_dists,        # 8
            car_dists,         # 8
            [velocity],       # 1
            [angle_to_target], # 1
            [dist_to_target],  # 1
            [checkpoint_progress],   # 1
            [car_angle],  # 1
            [1.0 if self.vel >= 0 else 0.0]  # 1
        ])
        # State tensor
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).to(self.device)
        
        # Decision 
        with torch.no_grad():
            action_idx = self.model(state_tensor.unsqueeze(0)).argmax().item()
            
        action_str = self.ACTIONS[action_idx]
        return action_str


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