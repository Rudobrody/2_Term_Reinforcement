import matplotlib 
matplotlib.use("Agg") # This line is blocking pop up windows and it jsut saved img
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import imageio
import os
import itertools
from enum import Enum, auto


# Define constants
# TARGET = 8
# GRID_SIZE = 3
SEED = 43


class Action(Enum):
    """Enumeration for Slider actions"""
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class Player(Enum):
    "Enumeration for the current player turn"
    SLIDER = auto() # auto automatically assigns a unique value to each Enum member
    SPAWNER = auto()


class Adversarial2048Env:
    """
    A 3x3 2048 Env designed for adversarial multi-agent game

    The game is modeled as a zero-sum turn-based game
    First Agent (slider): Slider is trying to merge number to reach the target.
    Second Agnt (Spawner): Spawner places a '2' or '4' tile in empty spots to block First Player
    """

    def __init__(self, grid_size, target):
        """Initializes the environment with a 3x3 grid"""
        self.grid_size = grid_size
        self.target = target
        self._rng = np.random.default_rng(SEED)

        # Initialization of board
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Track current player
        self.current_player = Player.SLIDER # First move has a Slider

        # Visualization cache
        # self.frames = []
        self.history = []

        # Define color map
        self.tile_colors = {
            0: "#000000", 2: "#FA4F00", 4: "#FAF603", 8: "#47FC00", 16: "#00FFFF", 32: "#CC00FF"
        }

        # Define possible values
        self.values = [0] + [2**i for i in range(1, int(np.log2(self.target)) + 1)]


    def get_all_states(self) -> list[tuple]:
        """Generates all possible 3x3 boards. So all combinations of value in every tail"""
        print("Generating state space..")
        
        # Making cartesian product, it should give us 4^9 = 262,144 states
        states = list(itertools.product(self.values, repeat=self.grid_size * self.grid_size))
        print(f"State space generated. The number of all states is equal: {len(states)}")
        return states
    

    def get_next_states(self, state, action):
        """
        Calculate probabilities to go to the next state
        
        Returns:
            tuple: {next_state_tuple: probability}
        """
        # Convert tuple state to numpy array
        current_board = np.array(state).reshape(self.grid_size, self.grid_size)

        # Simulate the slide
        slip_board, _ = self._apply_slide(action, board=current_board)

        # Simulate the spawn
        empties = list(zip(*np.where(slip_board == 0)))

        if not empties:
            return {tuple(slip_board.flatten()): 1.0}
        
        # Calculate the probabilities, numbers are spawned uniform
        transitions = {}
        prob = 1.0 / len(empties)

        # Simulate creating 2 in every empty space
        for r, c in empties:

            # Becuase we take a copy of slip board there always be only a one '2'
            candidate = slip_board.copy()
            candidate[r, c] = 2 # always will spawn '2'

            # We create a flattened array with this new '2'
            next_state_tuple = tuple(candidate.flatten())

            # This formula is boilerplate, because every state will be unique
            if next_state_tuple in transitions:
                transitions[next_state_tuple] += prob
            else:
                transitions[next_state_tuple] = prob

        return transitions


    def get_reward(self, state, action, next_state):
        """Retruns the reward for taking 'action' in 'state'"""
        # Convert to array
        current_board = np.array(state).reshape(self.grid_size, self.grid_size)

        # Simulate the slide to get the merge reward
        _, reward = self._apply_slide(action, board=current_board)

        # Add bonus for win if target is reached
        if self.target in next_state:
            reward += 100

        return reward


    def reset(self) -> np.array:
        """
        Reset the env to the initial state.

        Returns:
            np.ndarray: The initial board state
        """

        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        #self.frames = []
        self.history = []

        # Spawning two tiles randomly
        empty_cells = self._get_empty_cells()

        # If there is more than one empty cell
        if len(empty_cells) >= 2:

            # We choosee 2 indices from this list
            indices = self._rng.choice(len(empty_cells), 2, replace=False)

            for idx in indices:
                # row, column
                r, c = empty_cells[idx]
                self.board[r, c] = 2
            
        self.current_player = Player.SLIDER

        # Record initial state (t=0)
        self._record_history(self.board, self.board, "Start")

        return self.board.copy()
    

    def get_legal_moves(self, state=None) -> list[int]:
        """Returns a list of legac actions for the current player in a given state"""

        # If a state tuple is provided, convert it to a board array, QLearning need it to insepct future
        board_to_check = None
        if state is not None:
            board_to_check = np.array(state).reshape(self.grid_size, self.grid_size)
        else:
            board_to_check = self.board

        if self.current_player == Player.SLIDER:
            return self._get_slider_moves(board=board_to_check)
        elif self.current_player == Player.SPAWNER:
            return self._get_spawner_moves(board=board_to_check)


    def _get_spawner_moves(self, board: np.ndarray=None)-> list[int]:
        """Private heler: Returns all empty cell indices for specific board"""
        legal_moves = []

        # All empty cells are legal moves
        empties = self._get_empty_cells(board)

        for r, c in empties:
            legal_moves.append(r * self.grid_size + c)
        return legal_moves
    

    def _get_slider_moves(self, board:np.ndarray=None) -> list[int]:
        """
        Private helper: Checks which of the directions are valid
        If 'board' is provided, checks that. If none, checks self.board
        """
        # Determine which board is current
        current_board = self.board if board is None else board
        
        legal_moves = []

        # Checking left side
        can_left = False

        # We take columns 1 to n
        for r in range(self.grid_size):
            for c in range(1, self.grid_size):
                val = current_board[r, c]

                # If there is non zero element so it has to have a neigbor, even 0
                if val != 0:
                    neighbor = current_board[r, c-1]
                    if neighbor == 0 or neighbor == val:
                        can_left = True
                        break # If we found possible move left we don't check rest of rows
            if can_left: break
        if can_left: legal_moves.append(Action.LEFT.value)
        
        # Checking right side
        can_right = False

        # We take columns 0 to n-1 
        for r in range(self.grid_size):
            for c in range(0, self.grid_size - 1):
                val = current_board[r, c]

                if val != 0:
                    neighbor = current_board[r, c+1]
                    if neighbor == 0 or neighbor == val:
                        can_right = True
                        break 
            if can_right: break
        if can_right: legal_moves.append(Action.RIGHT.value)

        # Checking up
        can_up = False

        # We take rows 1 to n 
        for r in range(1, self.grid_size):
            for c in range(self.grid_size):
                val = current_board[r, c]

                if val != 0:
                    neighbor = current_board[r-1, c]
                    if neighbor == 0 or neighbor == val:
                        can_up = True
                        break 
            if can_up: break
        if can_up: legal_moves.append(Action.UP.value)

        # Checking down
        can_down = False

        # We take rows 0 to n-1 
        for r in range(self.grid_size - 1):
            for c in range(self.grid_size):
                val = current_board[r, c]

                if val != 0:
                    neighbor = current_board[r+1, c]
                    if neighbor == 0 or neighbor == val:
                        can_down = True
                        break 
            if can_down: break
        if can_down: legal_moves.append(Action.DOWN.value)
        
        return legal_moves


    def get_possible_actions(self, state):
        """Returns actions for slider for a hypothetical 'state' without modyfing self.board"""
        # Returns [] if the state is already a WIN state
        if self.target in state:
            return []
        
        # Convert tuple to an array
        hypothetical_board = np.array(state).reshape(self.grid_size, self.grid_size)

        # pass this board to get moves for slider
        return self._get_slider_moves(board=hypothetical_board)


    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, any]]:
        """
        Executes a step in the env based on the current player's turn.

        If turn is Slider:
            action (int): 0=Left, 1=Down, 2=Right, 3=Up

        if turn is Spawner:
            action (int): Flat index (0 to 8) of where to place a tile.

        Returns:
            new_board_state (np.ndarray)
            reward (float): Reward generated by the move
            game_over (bool): True if game over 
            info (dict): Contains player turn for the next step
        """
        # Capture state beofre move
        prev_board = self.board.copy()
        mover_name = self.current_player.name

        reward = 0.0
        game_over = False
        info = {}

        if self.current_player == Player.SLIDER:
            new_board, move_reward = self._apply_slide(action)

            # Update of env
            self.board = new_board
            reward = move_reward

            # Check win condition
            if np.any(self.board == self.target):
                game_over = True
                reward += 100 
                info['result'] = 'WIN'
            
            # Pass turn to Spawner
            else:
                self.current_player = Player.SPAWNER 
        
        elif self.current_player == Player.SPAWNER:
            # Action is integer 0-8
            r, c = divmod(action, self.grid_size) # divmode returns tuple x//y, x%y

            # Place a tile '2'
            self.board[r, c] = 2

            reward = 0

            if not self._can_slider_move():
                game_over = True
                reward = 100
                info['result'] = 'LOSS'

            # Pass turn back to Slider
            self.current_player = Player.SLIDER

        info['turn'] = self.current_player.name
        
        # Save the history
        self._record_history(prev_board, self.board, mover_name)

        return self.board.copy(), reward, game_over, info


    def _record_history(self, prev, curr, mover):
        """Stores copies of the board states"""
        self.history.append({
            "prev": prev.copy(),
            "curr": curr.copy(),
            "mover": mover
        })


    def _apply_slide(self, action: int, board: np.ndarray = None) -> tuple[np.ndarray, float]:
        """
        Simulates a slide action. Returns (new_board, reward). It accepts optional
        'board' argument for simulation of movement
        """
        if board is None:
            board_to_use = self.board
        else:
            board_to_use = board

        board_rot = np.copy(board_to_use)

        # Roate board sot he desired direction is always 'LEFT'
        rotation = 0
        # Because np.rot90 is CCW so if we rotate once this will simulate move UP
        if action == Action.DOWN.value: rotation = 3
        elif action == Action.RIGHT.value: rotation = 2
        elif action == Action.UP.value: rotation = 1

        # Rotate grid counter-clockwise to align move to LEFT
        board_rot = np.rot90(board_rot, k=rotation)

        # Process merging to the LEFT
        new_board_rot, reward = self._merge_left(board_rot)

        # Rotate back
        new_board = np.rot90(new_board_rot, k=-rotation)

        return new_board, reward
        

    def _merge_left(self, board: np.ndarray) -> tuple[np.array, float]:
        """Slides all rows to the left and merge"""
        new_board = np.zeros_like(board)
        total_reward = 0.0

        for r in range(self.grid_size):
            # Remove zeros from row
            row = board[r][board[r] !=0]

            # Merge
            new_row = []
            skip = False

            # For every cell in a row which is not zero
            for i in range(len(row)):

                # Skip the cell if merge occure
                if skip:
                    skip = False
                    continue
                
                # i+1 just checks is that last cell in a row
                if i+1 < len(row) and row[i] == row[i+1]:
                    doubled = row[i] * 2 # pow 2
                    new_row.append(doubled)
                    total_reward += doubled # Add a reward for move which is equal to score
                    skip = True # We merge 2 cell into one so the next one has to be skipped
                else: # if there is no merge or it is last cell just copy the value of the old board
                    new_row.append(row[i])

            # Fill with zeros
            new_row += [0] * (self.grid_size - len(new_row))
            new_board[r] = np.array(new_row)

        return new_board, total_reward


    def _get_empty_cells(self, board: np.ndarray = None) -> list[tuple[int, int]]:
        """Returns list of (row, col) for all empty cells on a specific board"""
        # Use the passed board if it exsits, otherwise use the real self.board
        current_board = self.board if board is None else board
        return list(zip(*np.where(current_board == 0)))
    

    def _can_slider_move(self) -> bool:
        """Checks if any move is possible for the slider - it has no possible move in case when there is no empty cell and there is no possible merges"""
        if len(self._get_empty_cells()) > 0:
            return True
        
        # Check for possible merges
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = self.board[r, c]

                # Check right
                if c+1 < self.grid_size and self.board[r, c+1] == val:
                    return True
                
                # Check down
                if r+1 < self.grid_size and self.board[r+1, c] == val:
                    return True
        
        return False
    

    def render_frame(self, board_prev, board_curr, mover_name):
        """Renders the current board and save the frame"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
        fig.suptitle(f"Turn {mover_name}", fontsize=14, fontweight="bold")
        self._draw_board(ax1, board_prev, "Before")
        self._draw_board(ax2, board_curr, "After")

        # Save to buffer for GIF

        # Renders the plot
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        
        # Reshape 
        w, h = fig.canvas.get_width_height()
        image = image.reshape((h, w, 4)) # 4 channels RGBA
    
        # Drop the alpha channel (transparency) to get RGB
        image = image[:, :, :3]

        #self.frames.append(image)

        plt.close(fig)

        return image


    def _draw_board(self, ax, board, title):
        ax.clear()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.axis('off')
        ax.invert_yaxis()
        ax.set_title(title)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = board[r, c]
                color = self.tile_colors.get(val, '#FFFFFF') # white color for values > 32

                # Draw tile
                rect = plt.Rectangle((c, r), width=1, height=1, facecolor=color, edgecolor='white', linewidth=2)
                ax.add_patch(rect)

                # Draw text
                if val > 0:
                    text_color = "#ffffff" if val > 4 else "#be8c59"
                    ax.text(c + 0.5, r + 0.5, str(val), ha='center', va='center', fontsize=20, fontweight='bold', color=text_color)

        
    def save_gif(self, history, filename='game_replay.gif'):
        """Saves the recorded frames as a GIF"""
        frames = []

        for step_data in history:
            frame = self.render_frame(step_data['prev'], step_data['curr'], step_data['mover'])
            frames.append(frame)

        if frames:
            imageio.mimsave(filename, frames, fps=0.2)
            print(f"GIF saved to {filename}")



if __name__ == "__main__":
    env = Adversarial2048Env(grid_size=3, target=16)
    board = env.reset()

    # Simulate a few random movers
    for _ in range(30):

        # Get legal moves, inside of function it is devided on cases 'current player'
        legal_moves = env.get_legal_moves()

        if not legal_moves:
            print("No legal moves")
            break

        # Chosing random action from legal moves
        action = np.random.choice(legal_moves)

        # Execute step
        board, reward, game_over, info = env.step(action)

        if game_over:
            print(f"Game over {info}")

    #env.render_frame()
    env.save_gif(env.history, "test_run.gif")
    