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
import copy
import pandas as pd

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
    """A node int MCTS for a two-player game"""
    def __init__(self, state, player_to_move, parent=None, parent_action=None):
        """
        Args:
            statee: Board state as a tuple
            player_to_move: Which player acts from this node 
            parent: Parent MCTS node
            parent_action: Action that led to this node
        """
        # The board state
        self.state = state

        self.player_to_move = player_to_move

        # Parent node
        self.parent = parent

        # List of children nodes
        self.children = []

        # Action that led to this state
        self.parent_action = parent_action

        # Counter, how many this node was visited
        self.visits = 0

        # Counter of rewards from slider's perspective
        self.total_value = 0

        # Actions we haven't expanded yet
        self.untried_actions = None

    
    def is_fully_expanded(self):
        """Returns Ture if all legal moves from this state have been created as children"""
        return self.untried_actions is not None and len(self.untried_actions) == 0
    

    def best_child(self, c_param=1.43):
        """
        Select the best child using Upper Confidence Bound, adjusted for the current player.
        Slider maximizes value, spawner minimizes it.
        
        Args:
            c_param (float): Exploration parameter, equal sqrt(2)

        Notes:
            We are applying formula 
        """
        best_score = float('-inf')
        best_children = []

        # Calculate it here once 
        log_visits = math.log(self.visits)

        for child in self.children:
            # Unvisited nodes get highest priority
            if child.visits == 0:
                ucb_score = float('inf') # UCB - upper confidence bound

            # else will get an average 
            else:
                avg_value = child.total_value / child.visits

                # If it's a spawner's turn, we want to minimize slider's value so we negate
                if self.player_to_move == Player.SPAWNER:
                    avg_value = -avg_value

                # Exploration term
                exploration = c_param * math.sqrt(log_visits / child.visits)
                ucb_score = avg_value + exploration

            # If current ucb score is higher than max 
            if ucb_score > best_score:
                best_score = ucb_score
                best_children = [child]

            # Append if its equal
            elif ucb_score == best_score:
                best_children.append(child)
        
        return np.random.choice(best_children)
    

    def best_action_by_visits(self):
        """Return the action of the most visited child (used for final decision)"""
        if not self.children:
            return None
        best_child = max(self.children, key=lambda c: c.visits)
        return best_child.parent_action
    
# %%
class AdversarialMCTS:
    """
    Monte Carlo Tree Search for adversarial game 2048.
    Can be used bu either Slider or Spawner.
    """
        
    def __init__(self, env, iterations: int=500, c_param=1.41, simulation_depth=50):
        """
        Initialization of instance of MCTSAgent 

        Args:
            env: An instace of Adeversarial2048Env
            iter (int): Number of MCTS iterations per move
            c_param (float): Exploration parameter for UCB
            simulation_depth: Max depth for random rollouts
        """
        self.env = env
        self.iterations = iterations
        self.c_param = c_param
        self.simulation_depth = simulation_depth


    def get_action(self, state_tuple, current_player):
        """
        Run MCTS and return the best action for the current player
        
        Args:
            state_tuple: Current board state
            current_player: Player.SLIDER or Player.SPAWNER 
        """
        # Check if the root is already a terminal 
        if self.env.target in state_tuple:
            return None # Game already won
        
        # Get legal actions for current player
        legal_actions = self._get_legal_actions(state_tuple, current_player)

        if not legal_actions:
            return None
        
        # Create a root node
        root = MCTSNode(state=state_tuple, player_to_move=current_player)
        
        # We update what actions we have to check
        root.untried_actions = list(legal_actions)

        # Run MCTS iterations
        for i in tqdm(range(self.iterations), desc="Iteration for one move"):
            # Selection 
            node = self._select(root)

            # Expansion, if the node isn't a goal leaf
            if not self._is_terminal(node.state):
                
                # And the note haven't been fully expanded
                if not node.is_fully_expanded():
                    node = self._expand(node)

            # Simulation (Rollout), so play to the end
            reward = self._simulate(node.state, node.player_to_move)

            # Backpropagation
            self._backpropagate(node, reward)

        return root.best_action_by_visits()
    

    def _get_legal_actions(self, state, player):
        """get legal actions for a player at a given state"""
        board = np.array(state).reshape(self.env.grid_size, self.env.grid_size)

        if player == Player.SLIDER:
            return self.env._get_slider_moves(board)
        elif player == Player.SPAWNER:
            empties = list(zip(*np.where(board == 0)))
            return [r * self.env.grid_size + c for r, c in empties]
        
    
    def _is_terminal(self, state):
        """Check if a state is terminal (win or no moves)"""
        if self.env.target in state:
            return True
        
        # Check if slider can move
        board = np.array(state).reshape(self.env.grid_size, self.env.grid_size)
        slider_moves = self.env._get_slider_moves(board)

        if not slider_moves:
            return True
        
        return False
    

    def _select(self, node):
        """
        Traverse until node that has untried actions or is terminal.
        """
        # In other words when we try every action for a considered node and
        # this node has a children then pick best children
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c_param)
        return node
    

    def _expand(self, node):
        """
        Pick one untried action, create a new child node and return it
        """
        # Lazy initialization 
        if node.untried_actions is None:
            node.untried_actions = list(self._get_legal_actions(node.state, node.player_to_move))

        # If there is no untired actions just return the same node
        if not node.untried_actions:
            return node
        
        # Pick an untried action
        action = node.untried_actions.pop()

        # Compute next state based on a player
        next_state, next_player = self._apply_action(node.state, action, node.player_to_move)

        # Create a new child node
        child = MCTSNode(
            state=next_state,
            player_to_move=next_player,
            parent=node,
            parent_action=action
            )

        # Initialize untried actions fro child
        child.untried_actions = list(self._get_legal_actions(next_state, next_player))

        node.children.append(child)

        return child
    

    def _apply_action(self, state, action, player):
        """
        Apply an action and return (next_state, next_player)
        This is deterministic for both players in adversarial mode.
        """
        board = np.array(state).reshape(self.env.grid_size, env.grid_size)

        if player == Player.SLIDER:
            # Apply slide
            new_board, _ = self.env._apply_slide(action, board)
            next_state = tuple(new_board.flatten())
            next_player = Player.SPAWNER
        
        elif player == Player.SPAWNER:
            # Place a tile
            r, c = divmod(action, self.env.grid_size)
            new_board = board.copy()
            new_board[r, c] = 2
            next_state = tuple(new_board.flatten())
            next_player = Player.SLIDER

        return next_state, next_player
    

    def _simulate(self, state, starting_player):
        """
        Run a random simulation from this state to the end of the game. 
        Return reward from slider perspective
        """
        current_state = state
        current_player = starting_player
        cumulative_reward = 0
        depth = 0

        while depth < self.simulation_depth:
            
            # Check terminal conditions
            if self.env.target in current_state:
                cumulative_reward += 100
                break
            
            # Check legal moves
            legal_actions = self._get_legal_actions(current_state, current_player)

            # If there is no possile actions it should be game over
            if not legal_actions:
                if current_player == Player.SLIDER:
                    cumulative_reward -= 100
                break 

            # As it was said, we play randomly to the end
            action = np.random.choice(legal_actions)

            # Get reward for slider moves
            if current_player == Player.SLIDER:
                board = np.array(current_state).reshape(self.env.grid_size, self.env.grid_size)
                _, merge_reward = self.env._apply_slide(action, board)
                cumulative_reward += merge_reward

            # Apply action
            current_state, current_player = self._apply_action(current_state, action, current_player)
            depth += 1

        return cumulative_reward



    def _backpropagate(self, node, reward):
        """update the node and all ancestors"""
        while node is not None:
            node.visits += 1
            node.total_value += reward # Reward from slider's perspective
            node = node.parent


def to_hashable(board_array: np.ndarray) -> tuple:
    """Convert numpy array to hashable tuple"""
    return tuple(board_array.flatten())


def play_game(env, slider_agent, spawner_agent, verbose=False):
    """
    Play a complete game between slider and spawner agents
    
    Args:
        env: Game environment
        slider_agent: Slider agent
        spawner_agent: Spawner agent
        verbose: print game process
    
    Returns
        result: 'WIN' or 'LOSS' from slider's perspective
        history: game history for visualization
    """
    board = env.reset()
    done = False
    move_count = 0

    while not done:
        state_tuple = to_hashable(board)

        # Slider turn
        if env.current_player == Player.SLIDER:
        
            action = slider_agent.get_action(state_tuple, Player.SLIDER)

            if action is None:
                if verbose:
                    print("Slider has no moves")
                return 'LOSS', env.history
            
            board, reward, done, info = env.step(action)
            move_count += 1

            if done:
                result = info.get('result', 'UNKNOWN')
                if verbose:
                    print(f"game over: {result}")
                return result, env.history
        
        elif env.current_player == Player.SPAWNER:
            if done:
                break

            # Spawner turn
            action = spawner_agent.get_action(state_tuple, Player.SPAWNER)

            if action is None:
                break

            board, reward, done, info = env.step(action)

            if done:
                result = info.get('result', 'UNKNOWN')
                if verbose:
                    print(f"Game over {result}")
                return result, env.history
            
    return 'UNKNOWN', env.history

# %%
# lets create some plots

def run_experiment(targets: list[int], iteration_counts: list[int], num_games_per_config: int=10) -> pd.DataFrame:
    """
    Run a grid of experiment

    Args:
        targets (list): list of game targets
        iterations_counts (list): List of MCTS iterations
        num_games_per_config (int): how many games to play for averaging results

    Returns:
        pd.DataFrame: Results table
    """
    results_data = []

    # Loop over targets 
    for target in targets:
        
        # Loop over iterations
        for iter in iteration_counts:
            wins = 0
            total_score = 0

            # Run N games to get averaging
            for _ in tqdm(range(num_games_per_config), desc=f'Iter {iter}'):
                # Initialization of env
                env = Adversarial2048Env(grid_size=3, target=target)

                # Init agents
                slider_mcts = AdversarialMCTS(env, iterations=iter)
                spawner_mcts = AdversarialMCTS(env, iterations=iter)

                # Play
                result, _ = play_game(env, slider_mcts, spawner_mcts)

                # Record 
                if result == 'WIN':
                    wins += 1
                
                # Calculate simple score (sum of board)
                total_score += np.sum(env.board)

            win_rate = wins / num_games_per_config
            avg_score = total_score / num_games_per_config

            results_data.append({
                'Target': target,
                'Iterations': iter,
                'Win Rate': win_rate,
                'Average Score': avg_score
            }) 

    return pd.DataFrame(results_data)


# %% 
# Plotting of results of our experiment

def plot_results(df):
    """Generates the family of plots"""

    # Win Rate vs iterations
    plt.figure(figsize=(10, 7))

    targets = df['Target'].unique()
    for target in targets:
        subset = df[df['Target'] == target]
        plt.plot(subset['Iterations'], subset['Win Rate'], marker='o', linewidth=2, label=f'target {target}')
    
    plt.title("Win rate vs iterations")
    plt.xlabel("Iterations per move")
    plt.ylabel("Win rate")
    plt.legend()
    plt.savefig("MCTS/mcts_win_rate.png")
    plt.show()

    # Average score vs iterations
    plt.figure(figsize=(10, 7))
    for target in targets:
        subset = df[df['Target'] == target]
        plt.plot(subset['Iterations'], subset['Average Score'], label=f'target {target}')

    plt.title("Average score vs iterations")
    plt.xlabel('Iterations per move')
    plt.ylabel('Average score')
    plt.legend()
    plt.savefig("MCTS/mcts_average_score.png")
    plt.show()

# %%


# %%
if __name__ == "__main__":
    # Configuration
    GRID_SIZE = 3
    TARGET = 32
    NUM_GAMES = 20
    MCTS_ITERATIONS = 200 

    # Create an environment
    env = Adversarial2048Env(grid_size=GRID_SIZE, target=TARGET)

    # Create agents
    slider_mcts = AdversarialMCTS(env, iterations=MCTS_ITERATIONS)
    spawner_agent = AdversarialMCTS(env, iterations=MCTS_ITERATIONS)

    # Play games and track results
    results = {'WIN': 0, 'LOSS': 0, 'UNKNOWN': 0}

    for game_num in tqdm(range(NUM_GAMES), desc='games'):
        result, history = play_game(env, slider_mcts, spawner_agent, verbose=False)
        results[result] += 1

        # Save first game as GIF
        if game_num == 0:
            env.save_gif(history, f"MCTS/MCTS_game_t{TARGET}.gif")

    print(f"Slider wins: {results['WIN']}")
    print(f"Slider Loses: {results['LOSS']}")

    experiment_targets = [32, 64, 128, 256, 512]
    experiment_iters = [50, 100, 200]

    # Run
    df_results = run_experiment(targets=experiment_targets, iteration_counts=experiment_iters, num_games_per_config=5)

    print("Experiment results")
    print(df_results)

    plot_results(df_results)

