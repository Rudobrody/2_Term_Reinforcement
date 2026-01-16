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

    # Convert to hashable tuple for the Agent
    state = to_hashable(board)

    done = False

    while not done:

        # Slider turn
        if env.current_player == Player.SLIDER:
            
            # get agent to pick action given state state.
            action = slider_agent.get_action(state)

            # Checking game over
            if action is None: 
                
                # Give him a punishment 
                slider_agent.update(state, 0, -100, state) # dummy 0, placeholder
                
                # And give also a big reward for spawner
                if last_spawner_state:
                    spawner_agent.update(last_spawner_state, last_spawner_action, 100, state)
                break

            # Make a step
            next_state, slider_reward, done, _ = env.step(action)
            next_state = to_hashable(next_state)
              
            # Update tell me how good is to take action "a" in a state "s"
            slider_agent.update(state, action, slider_reward, next_state)

            # Update spawner
            if last_spawner_state is not None:

                # Logic is simple, spawner will get inverse reward of slider
                spawner_reward = -slider_reward
                spawner_agent.update(last_spawner_state, last_spawner_action, spawner_reward, next_state)

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
            next_state, spawn_reward, done, _ = env.step(spawner_action)

            # Update state, so spawner could see a new tile
            next_state = to_hashable(next_state)

            # Store memory (wait for slider's reation before updating)
            last_spawner_state = state
            last_spawner_action = action

            # Updating of next state
            state = next_state

    return total_reward

# %%
env = Adversarial2048Env()
agent = QLearningAgent(alpha=0.1, epsilon=0.1, discount=0.99,
                       get_legal_actions=env.get_possible_actions)

rewards = []
for i in tqdm(range(1000), desc="Training episodes"):
    rewards.append(play_and_train(env, agent))
    

plt.plot(rewards)
plt.ylabel('Rewards')
plt.xlabel('Number of episode')
plt.savefig('training_curve.png')

# %%
