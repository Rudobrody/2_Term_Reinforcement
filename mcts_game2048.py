# %%
"""
Every time before move algorithm go through the loop thousand of times
The loop consists from:
- Selection (start at the root), traverse down picking the most promising nodes
using Upper confidence bound until it hits a state we haven't explored yet
- Expansion: If the leaf node isn't game-over state we will add one or more
child nodes to the three (create the next possible states)
- Simulation - from the new child node we play a completely random game until
end of episode
- backpropagation - we take the result of that random and update the statistics
of all nodes that were passed to get there
"""

# %% 
import math
from game2048 import Action, Player, Adversarial2048Env
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% [markdown]
# In MCTS the main problem was to select child nodes and find balance
# between exploration and chosing nodes with best win rate. The formula   
# which is trying to cover that problem is:  

# %% [markdown]
# \begin{equation}
# \frac{w_i}{n_i} + c * \sqrt{\frac{\ln(N_i)}{n_i}}
# \end{equation}
# where:
# w_i - number of wins for the node  
#
# n_i - number of simulation for the node  
#
# N_i - total number of simulations run by the parent node of the one considered  
#
# c - exploration parameter  

# %%
class MCTSNode:
    def __init__(self, state, parent=None, parent_action=None):
        # The board state
        self.state = state

        # Parent node
        self.parent = parent

        # List of children nodes
        self.children = []

        # Action that led to this state
        self.parent_action = parent_action

        # Counter, how many this node was visited
        self.visits = 0

        # Counter of rewards
        self.value = 0

        # Actions we haven't expanded yet
        self.untried_actions = []

    
    def is_fully_expanded(self):
        """Returns Ture if all legal moves from this state have been created as children"""
        return len(self.untried_actions) == 0
    

    def best_child(self, c_param=1.43):
        """
        Select the best child using Upper Confidence Bound
        
        Args:
            c_param (float): Exploration parameter, equal sqrt(2)

        Notes:
            We are applying formula 
        """
        choice_weights = [
            (c.value / c.visits) + c_param * np.sqrt((np.log(self.visits) / c.visits))
            for c in self.children
        ]
        return self.children[np.argmax(choice_weights)]
    
# %%
class MCTSAgent:
    """
    Monte Carlo Tree Search Agent for game 2048
    """
        
    def __init__(self, env, iter: int=100):
        """
        Initialization of instance of MCTSAgent 

        Args:
            env: An instace of Adeversarial2048Env
            iter (int): Number of iterations per move
        """
        self.env = env
        self.iter = iter


    def get_action(self, current_board_tuple):
        """Returns the best action for the given board state"""
        # Create a root node
        root_node = MCTSNode(state=current_board_tuple)

        # Check if the root is already a terminal 
        if self.env.target in current_board_tuple:
            return None
        
        # Find legal moves 
        legal_actions = self.env.get_possible_actions(current_board_tuple)

        # We update what actions we have to check
        root_node.untried_actions = legal_actions

        for i in tqdm(range(self.iter), desc="Iteration for one move"):
            # Selection 
            node = self._select(root_node)

            # Expansion, if the node isn't a goal leaf
            if not self.env.target in node.state:
                
                # And the note haven't been fully expanded
                if not node.is_fully_expanded():
                    node = self._expand(node)

            # Simulation (Rollout), so play to the end
            reward = self._simulate(node.state)

            # Backpropagation
            self._backpropagate(node, reward)

        # Check is the node has a children
        if not root_node.children:
            return np.random.choice(legal_actions) if legal_actions else None

        # Return best move - most visited
        best_node = sorted(root_node.children, key=lambda c: c.visits)[-1]

        return best_node.parent_action
    

    def _select(self, node):
        """
        Traverse until node that has untried actions or is terminal.
        """
        # In other words when we try every action for a considered node and
        # this node has a children then pick best children
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node
    

    def _expand(self, node):
        """
        Pick one untried action, create a new child node and return it
        """
        action = node.untried_actions.pop()

        # Determine the next state, transition is implemented in env, probability
        # is equal (spawn tile)
        transitions = self.env.get_next_states(node.state, action)

        # Sample a next stated based on probabilities
        states = list(transitions.keys())
        probs = list(transitions.values())
        next_state_idx = np.random.choice(len(states), p=probs)
        next_state = states[next_state_idx]

        # Create a new child node
        child_node = MCTSNode(state=next_state, parent=node, parent_action=action)

        # If next state is our goal 
        if self.env.target in next_state:
            child_node.untried_actions = []
        else:
            # Initialize untried actions for this new child
            child_legal_moves = self.env.get_possible_actions(next_state)
            child_node.untried_actions = child_legal_moves

        # Update that a node has a children
        node.children.append(child_node)

        return child_node
    

    def _simulate(self, state):
        """
        Run a random simulation from this state to the end of the game. 
        Return a final reward
        """
        current_sim_state = state
        cumulative_reward = 0
        depth = 0

        # Depth is limited to prevent infinite loop in bad policies
        max_depth = 20

        while depth < max_depth:
            if self.env.target in current_sim_state:
                cumulative_reward += 100
                break
            
            # Check legal moves
            possible_actions = self.env.get_possible_actions(current_sim_state)

            # If there is no possile actions it should be game over
            if not possible_actions:
                break 

            # As it was said, we play randomly to the end
            action = np.random.choice(possible_actions)

            # Get next states
            transitions = self.env.get_next_states(current_sim_state, action)

            # Sample a next stated based on probabilities
            states = list(transitions.keys())
            probs = list(transitions.values())
            next_state_idx = np.random.choice(len(states), p=probs)
            next_state = states[next_state_idx]

            # getting reward
            reward = self.env.get_reward(current_sim_state, action, next_state)
            cumulative_reward += reward


            current_sim_state = next_state
            depth += 1
            
        return cumulative_reward


    def _backpropagate(self, node, reward):
        """update the node and all ancestors"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


# %%
# Creating helper to convert a numpy array to a hashable tuple
def to_hashable(board_array: np.ndarray) -> tuple:
    """Helper which converts a numpy array to a hashable tuple"""
    return tuple(board_array.flatten())

# %%
# Play
env = Adversarial2048Env()
mcts = MCTSAgent(env, iter=500)

board = env.reset()
done = False

while not done:
    if env.current_player == Player.SLIDER:
        state_tuple = to_hashable(board)

        # Search for best action
        action = mcts.get_action(state_tuple)

        # If returned action is None it means that slider has no avaiable moves
        if action is None:
            print("MCTS Resigned")
            break

        # Making a step
        board, reward, done, info = env.step(action)

    elif env.current_player == Player.SPAWNER:
        # If it is a game over
        if done: break

        spawner_moves = env.get_legal_moves()

        if not spawner_moves:
            break

        spawner_action = np.random.choice(spawner_moves)

        # Execute
        board, reward, done, info = env.step(spawner_action)

print(f"Game over. Result: {info}")

# %%
