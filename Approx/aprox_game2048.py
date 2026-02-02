# %% 
# Import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from game2048 import Player, Action, Adversarial2048Env

# %%
# Setup device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Used device: {device}")

# %%
class ReplayBuffer:
    """
    Experience Replay Buffer for DQN
    Stores transitions (state, action, reward, next_state, done) and samples them randomly.
    """
    def __init__(self, capacity: int):
        """
        Args:
            capacity (int): Max number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)

    
    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch_size: int):
        """
        Randomly sample a batch of transitions.

        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones)
        """

        # Get a random batch
        batch = random.sample(population=self.buffer, k=batch_size)

        # Unzip the batch into separate lists
        state, action, reward, next_state, done = zip(*batch)

        # Convert to pytorch tensors and move them to device
        return (
            torch.tensor(np.array(state), dtype=torch.float32).to(device),
            torch.tensor(np.array(action), dtype=torch.long).to(device),
            torch.tensor(np.array(reward), dtype=torch.float32).to(device),
            torch.tensor(np.array(next_state), dtype=torch.float32).to(device),
            torch.tensor(np.array(done), dtype=torch.float32).to(device),
        )
    

    def __len__(self):
        return len(self.buffer)
    
# %%
class DQN(nn.Module):
    """
    Deep Q-Network
    A simple MLP that maps board state -> action values
    """
    def __init__(self, grid_size: int, n_actions: int):
        """
        Args:
            grid_size (int): Size of the gird
            n_actions (int): Number of possible actions
        """
        super(DQN, self).__init__()

        self.grid_size = grid_size
        input_dim = grid_size * grid_size

        # Neural Network Architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Batch of board states.
        """
        return self.net(x)
    

    def preprocess_state(self, board: np.ndarray) -> np.ndarray:
        """
        Normalization - values are converted to thei log2 representation 
        
        Args:
            board (np.ndarray): Raw board
        
        Returns:
            np.ndarray: Flattened, log-normalized float array
        """
        # Replace 0 with 1 to avoid log(0)
        board_flat = board.flatten()
        board_log = np.where(board_flat > 0, np.log2(board_flat), 0.0)

        return board_log.astype(np.float32)

# %%
class DQNAgent:
    def __init__(self, grid_size, n_actions=4, epsilon=1.0, alpha=0.001, gamma=0.99, buffer_size=10000):
        """
        Args:
            grid_size (int): Size of board
            n_actions (int): Numer of actions
            epsilon (float): Exploration rate
            alpha (float): Learning rate for Optimizer
            gamma (float): Discount factor
            buffer_size (int): The size of the memory storage
        """
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = ReplayBuffer(capacity=buffer_size)

        # Networks
        # Policy network is the active network, so its decides actions and get updated ever single step
        self.policy_net = DQN(grid_size, n_actions).to(device)

        # Target net is frozen copy. It is used to calculate future max value)
        self.target_net = DQN(grid_size, n_actions).to(device)

        # Synchronize networks initially
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Changig mode to eval
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.criterion = nn.SmoothL1Loss()

    def get_action(self, state_tuple, legal_moves_indices=None):
        """
        Args:
            state_tuple: Raw board tuple
            legal_moves_indices (list): Optional list of legal ction indices. if provided masking is applied
        """
        
        # Exploration
        if random.random() < self.epsilon:
            if legal_moves_indices:
                return random.choice(legal_moves_indices)
            return random.randint(0, self.n_actions - 1)
        
        # Exploitation
        with torch.no_grad():

            # Normalization
            state_array = self._preprocess(state_tuple)
            state_tensor = torch.tensor(state_array, dtype=torch.float32, device=device).unsqueeze(0)

            q_values = self.policy_net(state_tensor)

            # Masking illeggal moves 
            if legal_moves_indices is not None:
                
                # Create a mask with -inf
                mask = torch.full_like(q_values, float('-inf'))
                
                # Unmask legal moves
                mask[0, legal_moves_indices] = 0 

                q_values = q_values + mask

            return q_values.argmax().item()
        

    def update(self, batch_size=64):
        
        if len(self.memory) < batch_size:
            return    
        
        # Sample
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Predict Q(s, a)
        # It gathers the specific Qvalue for the action actually taken
        # self.policy_net(states) -> [BatchSize, 4], values for all 4 actions
        # gather(1, actions.unsqueeze(1)) -> for each game in the batch, pick the Q-value of the action we actually tok
        # as a result there is a list of predictions, one for each experience in memory
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Predict target Q(s', a')
        with torch.no_grad():

            # self.target(next_states) -> [BatchSize, 4] for the next board state and then we choose best score
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Loss and Backpropagation
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    
    def sync_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    
    def _preprocess(self, state_tuple):
        """Log normalization of the board"""
        board = np.array(state_tuple).reshape(self.grid_size, self.grid_size)
        flat = board.flatten()

        return np.where(flat > 0, np.log2(flat), 0.0).astype(np.float32)
    

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

# %%
# Training loop
def play_and_train_dqn(env, slider, spawner, batch_size=64):
    board = env.reset()

    # State must be hashable for history
    state = tuple(board.flatten())

    last_spawner_state = None
    last_spawner_action = None
    last_spawner_bonus = 0

    total_reward = 0
    winner = None
    done = False

    while not done:

        # Slider turn
        if env.current_player == Player.SLIDER:

            # Get legal moves
            legal_indices = env.get_possible_actions(state)

            if not legal_indices:

                # Slider dies because of no moves and we have to store death experience
                processed_state = slider._preprocess(state)
                slider.memory.push(processed_state, 0, -100, processed_state, 1)

                # And of course update spawner - winner
                if last_spawner_state is not None:
                    spawner.memory.push(spawner._preprocess(last_spawner_state), last_spawner_action, 100, spawner._preprocess(state), 1)
                
                winner = 'Spawner'
                break

            # Select action
            action = slider.get_action(state, legal_indices)

            # Step
            next_board, reward, done, info = env.step(action)
            next_state = tuple(next_board.flatten())

            # Check win
            if done and info.get('result') == 'WIN':
                winner = 'Slider'

            # Store slider experience
            slider.memory.push(slider._preprocess(state), action, reward, slider._preprocess(next_state), 1 if done else 0)

            # Train slider
            slider.update(batch_size)

            # Store spawner experience (delayed reward)
            if last_spawner_state is not None:
                # Add heuristic penalty for max tile increase
                max_tile_penalty = 0
                curr_max = np.max(np.array(next_state))
                prev_max = np.max(np.array(last_spawner_state))
                if curr_max > prev_max: max_tile_penalty = -50

                spawner_reward = -reward + last_spawner_bonus + max_tile_penalty

                spawner.memory.push(spawner._preprocess(last_spawner_state), last_spawner_action, spawner_reward, spawner._preprocess(next_state), 0)

                # Train spawner
                spawner.update(batch_size)

            # Update
            state = next_state
            total_reward += reward

        # Spawner turn
        elif env.current_player == Player.SPAWNER:
            legal_indices = env.get_legal_moves(state)

            # If board is full
            if not legal_indices: break 

            action = spawner.get_action(state, legal_indices)

            next_board, _, done, info = env.step(action)
            next_state = tuple(next_board.flatten())

            # Check if the spawner won (slider has no moves)
            if done and info.get('result') == "LOSS":
                winner = 'Spawner'
                spawner.memory.push(spawner._preprocess(next_state), action, 100, spawner._preprocess(state), 1)
            else:
                # Calculate suffocation bonus
                slider_moves_count = len(env._get_slider_moves(
                    np.array(next_state).reshape(env.grid_size, env.grid_size)
                ))
                suffocation_bonus = (4 - slider_moves_count) * 10        

            # Store memory (wait for slider's reation before updating)
            last_spawner_state = state
            last_spawner_action = action
            last_spawner_bonus = suffocation_bonus

            # Updating of next state
            state = next_state
    
    return total_reward, winner

# %%
def run_dqn_showcase(env, slider, spawner, filename="aprox_showcase.gif"):
    """
    Runs a single game with Epsilon=0 (Pure Inference) and saves a GIF.
    """
    print(f"Generating Showcase: {filename}...")
    
    # Turn off Exploration
    slider.turn_off_learning()
    spawner.turn_off_learning()
    
    # Reset
    board = env.reset()
    state = tuple(board.flatten())
    done = False
    
    while not done:

        # Slider
        if env.current_player == Player.SLIDER:

            # Get only valid moves for the neural network mask
            legal_indices = env.get_possible_actions(state)
            if not legal_indices: break
            
            action = slider.get_action(state, legal_indices)
            next_board, _, done, info = env.step(action)
            
            if done and info.get('result') == 'WIN':
                print(f"Result: Slider WON ")
            
        # Spawner
        elif env.current_player == Player.SPAWNER:
            legal_indices = env.get_legal_moves(state)
            if not legal_indices: break
            
            action = spawner.get_action(state, legal_indices)
            next_board, _, done, info = env.step(action)
            
            if done and info.get('result') == 'LOSS':
                print(f"Result: Spawner WON")
                
        state = tuple(next_board.flatten())

    # Save
    env.save_gif(env.history, filename)

# %%
# main 
if __name__ == "__main__":
    TARGETS = [32, 64, 128, 256]
    GRID_SIZE = 3
    EPISODES = 3000
    SYNC_FREQ = 100 # After how many episodes there should be sync of target network

    # Store all results
    all_results = {}
    plt.figure(figsize=(10,7))

    for target in TARGETS:
        print(f"Training, target: {target}")
        env = Adversarial2048Env(grid_size=GRID_SIZE, target=target)

        # Initialize Agents
        slider = DQNAgent(GRID_SIZE, n_actions=4, epsilon=1.0)
        spawner = DQNAgent(GRID_SIZE, n_actions=9, epsilon=1.0)

        winners = []

        print(f"Starting DQN trainig with target: {target}")

        for episode in tqdm(range(EPISODES), desc=f'target:{target}'):
            _, winner = play_and_train_dqn(env, slider, spawner)
            winners.append(1 if winner == 'Slider' else 0)

            # Decay epsilon
            slider.decay_epsilon()
            spawner.decay_epsilon()

            # Sync target networks periodically 
            if episode % SYNC_FREQ == 0:
                slider.sync_target_network()
                spawner.sync_target_network()

        # Plot
        series = pd.Series(winners).rolling(100, min_periods=1).mean()
        all_results[target] = series

        # Add line to plot
        plt.plot(series, label=f'target: {target}')

        # Show case 
        if target == 32:
            if series.iloc[-1] > 0.1:
                run_dqn_showcase(env, slider, spawner, filename='aprox_showcase.gif')
   
    
    plt.axhline(0.5, linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel("Win rate")
    plt.legend()
    plt.savefig("aprox_win_rate.png")
    print(f"Final Win Rate: {series.iloc[-1]:.2f}")

# %%