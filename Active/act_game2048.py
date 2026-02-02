# %%
# Import env
from game2048 import Action, Player, Adversarial2048Env
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
# Taking implementation of QLearning Algorithm

class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        #
        # INSERT CODE HERE to get maximum possible value for a given state
        #
        qvalues = [self.get_qvalue(state, action) for action in possible_actions]     

        return max(qvalues)

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        #
        # INSERT CODE HERE to update value for the given state and action
        #
        value = (1 - learning_rate) * self.get_qvalue(state, action) + learning_rate * (reward + gamma * self.get_value(next_state))
        self.set_qvalue(state, action, value)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #
        # INSERT CODE HERE to get best possible action in a given state (remember to break ties randomly)
        #

        # Search for best Q(s,a)
        max_q_value = self.get_value(state)

        # Get all actions that have the maximum Q-value
        best_actions = [action for action in possible_actions if self.get_qvalue(state, action) == max_q_value]
        
        # Choose random action if there will be more than one with the highest score
        best_action = np.random.choice(best_actions)

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        #
        # INSERT CODE HERE to get action in a given state (according to epsilon greedy algorithm)
        #        

        # Epsilon is like an edge, if the drawn value is higher than epislon than we choose best ation, but if it's lower then we take random action
        
        # Picking a number from 0, 1 
        drawn_value = np.random.uniform(low=0, high=1)

        # Comparing to my edge - epsilon 
        if drawn_value > epsilon:

            # Choose best action
            chosen_action = self.get_best_action(state)
        
        # Otherwise, pick a random action
        else:
            chosen_action = np.random.choice(possible_actions)    
        
        return chosen_action

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0


# %%
# Creating helper to convert a numpy array to a hashable tuple
def to_hashable(board_array: np.ndarray) -> tuple:
    """Helper which converts a numpy array to a hashable tuple"""
    return tuple(board_array.flatten())

# %%
# Play and train implementation
def play_and_train(env, slider_agent, spawner_agent):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    board = env.reset()

    # Memory to punish the spawner later, because spawner has to wait for move of slider
    last_spawner_state = None
    last_spawner_action = None
    last_spawner_bonus = 0

    # Convert to hashable tuple for the Agent
    state = to_hashable(board)

    done = False

    # track the winner
    winner = None

    while not done:

        # Slider turn
        if env.current_player == Player.SLIDER:
            
            # get agent to pick action given state state.
            action = slider_agent.get_action(state)

            # Checking game over
            if action is None: 
                # print("DEBUG: Slider resigned")
                # Give him a punishment 
                slider_agent.update(state, 0, -100, state) # dummy 0, placeholder
                
                # And give also a big reward for spawner
                if last_spawner_state is not None: # Was without is not None
                    spawner_agent.update(last_spawner_state, last_spawner_action, 100, state)

                winner ='Spawner'
                break

            # Make a step
            next_state, slider_reward, done, info = env.step(action)

            # # Check if slider won this turn
            # if done and info.get('result') == 'WIN':
            #     print("DEBUG: Win registered")
            #     winner = 'Slider'

            next_state = to_hashable(next_state)
              
            # Update tell me how good is to take action "a" in a state "s"
            slider_agent.update(state, action, slider_reward, next_state)

            # Update spawner
            if last_spawner_state is not None:

                # Logic is simple, spawner will get inverse reward of slider and a bonus for suffocation
                # If the slider increased a max value of a tile then we have to punish spawner heavily
                max_tile_penalty = 0
                # if np.max(next_state) > np.max(np.array(last_spawner_state).reshape(3,3)):
                current_max = np.max(np.array(next_state).reshape(env.grid_size, env.grid_size))
                prev_max = np.max(np.array(last_spawner_state).reshape(env.grid_size, env.grid_size))
                if current_max > prev_max:
                    max_tile_penalty = -50
                
                total_spawner_reward = -slider_reward + last_spawner_bonus + max_tile_penalty
                spawner_agent.update(last_spawner_state, last_spawner_action, total_spawner_reward, next_state)

                # Reset memory after update
                last_spawner_state = None
                last_spawner_action = None

            if done and info.get('result') == 'WIN':
                #print("DEBUG: Win registered")
                winner = 'Slider'

            # Updating of next state
            state = next_state
            total_reward += slider_reward
        
        # Spawner turn
        elif env.current_player == Player.SPAWNER:
            
            if done: break

            # Get action for spawner, he sees the same state but has different actions
            spawner_action = spawner_agent.get_action(state)
            
            # If board is full - there is no place to spawn 
            if spawner_action is None: break

            # make a step, spawn reward will be almost always 0 because there is no points for spawning 2, but there is spawner reward after move of slider
            next_state, spawn_reward, done, info = env.step(spawner_action)

            # Update state, so spawner could see a new tile
            next_state = to_hashable(next_state)

            # Check if the spawner won (slider has no moves)
            if done and info.get('result') == "LOSS":
                winner = 'Spawner'
                spawner_agent.update(state, spawner_action, 100, next_state)
            else:
                # Calculate suffocation bonus
                slider_moves_count = len(env._get_slider_moves(
                    np.array(next_state).reshape(env.grid_size, env.grid_size)
                ))
                suffocation_bonus = (4 - slider_moves_count) * 10           

            # We can add heuristic reward, check how many moves the Slider has available now, so temporarily switch context to check legal moves for Slider
            # env.current_player = Player.SLIDER
            # slider_moves_count = len(env.get_legal_moves(next_state))
            
            # Switch back 
            # env.current_player = Player.SPAWNER 
            
            # Add a bonus for spawner is slider has fewer moves
            # suffocation_bonus = (4 - slider_moves_count) * 10

            # Store memory (wait for slider's reation before updating)
            last_spawner_state = state
            last_spawner_action = spawner_action
            last_spawner_bonus = suffocation_bonus

            # Updating of next state
            state = next_state

    return total_reward, winner


# %%
# function to display a showcase game
def run_showcase(env, slider_agent, spawner_agent, filename='Active/active_showcase.gif'):
    """Function to run a showcase game between slider and spawner agents"""
    # turn off Epxloration and Learning
    # We save original values just in case we wanted to continiue training later
    slider_eps, slider_alpha = slider_agent.epsilon, slider_agent.alpha 
    spawner_eps, spawner_alpha = spawner_agent.epsilon, spawner_agent.alpha

    slider_agent.turn_off_learning() # Setting alpha and epislon as 0
    spawner_agent.turn_off_learning()

    # Reset env
    board = env.reset()
    state = to_hashable(board)
    done = False

    print("Playing showcase game")

    while not done:
        # Slider turn
        if env.current_player == Player.SLIDER:
            # Pure greedy move
            action = slider_agent.get_action(state)

            if action is None:
                print("Slider has no moves")
                break

            next_board, reward, done, info = env.step(action)
            state = to_hashable(next_board)

            if done and info.get('result') == 'WIN':
                print('Slider won')
        
        # Spawner turn
        elif env.current_player == Player.SPAWNER:
            if done: break

            action = spawner_agent.get_action(state)

            if action is None: break

            next_board, reward, done, info = env.step(action)
            state = to_hashable(next_board)

            if done and info.get('result') == "LOSS":
                print("Spawner won")
    env.save_gif(env.history, filename)

    # Restore parameters
    slider_agent.epsilon, slider_agent.alpha = slider_eps, slider_alpha
    spawner_agent.epsilon, spawner_agent.alpha = spawner_eps, spawner_alpha



# %%
TARGETS = [8, 16, 32, 64, 128]
for target in TARGETS:

    # Create environment
    env = Adversarial2048Env(grid_size=3, target=target)

    #print(f"Debug: Created env, target {target} but actual env target {env.target}")
    # Create Slider agent
    slider = QLearningAgent(alpha=0.1, epsilon=0.1, discount=0.99,
                        get_legal_actions=env.get_possible_actions) # It returns [up, down]

    # Create Spawner agent
    spawner = QLearningAgent(alpha=0.1, epsilon=0.1, discount=0.9, get_legal_actions=env.get_legal_moves)

    # Store for showcase
    if target == 128:
        env_128 = env
        slider_128 = slider
        spawner_128 = spawner

    # Train loop
    rewards = []
    winners = []
    print('Training started')
    


    for i in tqdm(range(5000), desc=f"Training episodes for target {target}"): # Was 5000 -> 1000
        reward, winner = play_and_train(env, slider, spawner)
        rewards.append(reward)
        winners.append(1 if winner == 'Slider' else 0)
        
    win_rate_series = pd.Series(winners).rolling(window=100).mean()

    plt.figure(figsize=(10,6))
    plt.plot(win_rate_series, label='Slider win rate')

    # Adding benchmark
    plt.axhline(y=0.5, linestyle='--')
    plt.title(f'Slider vs Spawner, target={target}')
    plt.ylabel('Win rate')
    plt.xlabel('Number of episode')
    plt.legend()
    plt.savefig(f'Active/win_rate_t{target}.png')

# %%
run_showcase(env_128, slider_128, spawner_128)

