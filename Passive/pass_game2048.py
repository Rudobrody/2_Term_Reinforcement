# %%
# Import env
from game2048 import Action, Player, Adversarial2048Env
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Creating env
env_2048 = Adversarial2048Env(grid_size=3, target=8)
board = env_2048.reset()

# Generating all states
states = env_2048.get_all_states()
#print(states)

for s in states:
    actions = env_2048.get_possible_actions(s)
    for a in actions:
        next_states = env_2048.get_next_states(s, a)
        #print("State: " + str(s) + " action: " + str(a) + " " + "list of possible next states: ", next_states)



# %%
# Value iteration
def value_iteration(env, gamma, theta):
    """
            This function calculate optimal policy for the specified env using Value Iteration approach:

            'env' - model of the environment, use following functions:
                get_all_states - return list of all states available in the environment
                get_possible_actions - return list of possible actions for the given state
                get_next_states - return list of possible next states with a probability for transition from state by taking
                                  action into next_state
                get_reward - return the reward after taking action in state and landing on next_state


            'gamma' - discount factor for env
            'theta' - algorithm should stop when minimal difference between previous evaluation of policy and current is
                      smaller than theta
            Function returns optimal policy and value function for the policy
       """
    V = dict()
    policy = dict()

    all_states = env.get_all_states()

    # init with a policy with first avail action for each state
    for current_state in all_states:
        V[current_state] = 0

        # Get actions for this specific state
        possible_actions = env.get_possible_actions(current_state)
        
        if possible_actions:
            policy[current_state] = possible_actions[0]
        else:
            policy[current_state] = None

    #
    # INSERT CODE HERE to evaluate the best policy and value function for the given mdp
    #
    
    while True:
        delta = 0
        
        # For each state
        for state in all_states:
            # Skip terminal states
            if not env.get_possible_actions(state):
                continue
                
            # Store old value
            old_v = V[state]
            
            # Initialize for best action search
            best_action = None
            best_value = float('-inf')
            
            # Look at all possible actions
            for action in env.get_possible_actions(state):
                
                # Calculate value for this action
                action_value = 0
                
                # Consider all possible next states
                for next_state, prob in env.get_next_states(state, action).items():
                    reward = env.get_reward(state, action, next_state)
                    action_value += prob * (reward + gamma * V[next_state])
                
                # Update best action if this one is better
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            
            # Update value function with best value found
            V[state] = best_value
            # Update policy with best action
            policy[state] = best_action
            
            # Update delta for convergence check
            delta = max(delta, abs(old_v - best_value))
        
        # Check if we've converged
        if delta < theta:
            break

    return policy, V

# %%
# Creating a function for heatmap
def visualize_strategic_preference(value_dict, grid_size=3):
    """
    Visualize where the agent prefer to keep its highest tile
    
    Args:
        value_dict: 
        grid_size (int): Size of the board
    """
    # Creating accumulators for the heatmap
    
    # Value_sum sotres the total V for states where the max tile is at (row, column)
    value_sum = np.zeros((grid_size, grid_size))

    # Counts stores how many such states exist
    counts = np.zeros((grid_size, grid_size))

    for state_tuple, val in value_dict.items():
        # Conver tuple to array to find max tile location
        board = np.array(state_tuple).reshape(grid_size, grid_size)

        # Find coordinates of the maximum tile
        max_idx = np.argmax(board)

        # Because argmax gives flat index we use unravel_index which converts to (row, column)
        row, column = np.unravel_index(max_idx, (grid_size, grid_size))

        value_sum[row, column] += val
        counts[row, column] += 1

    # Calculate the average value per cell, thanks that we add argument 'where' there will be no error with division by zero
    # Why we define out? because there must be information what value put when there is division by zero
    avg_value_map = np.divide(value_sum, counts, out=np.zeros_like(value_sum), where=counts!=0)

    # Plot
    plt.figure(figsize=(8,6))
    sns.heatmap(avg_value_map, annot=True, cbar_kws={'label': 'Average Value'}) # Annot prints actual numerical value inside each cell
    plt.title(f"Agent strategy: Max tile preferred place", fontsize=14, fontweight='bold')
    plt.xlabel('column')
    plt.ylabel('Row')

    # Saving
    plt.savefig('Passive/Heatmap_passive.png')


# %%
optimal_policy, optimal_value = value_iteration(env_2048, 0.9, 0.001)

# %%
print(f"Optimal policy is: {optimal_policy}\n\n")
print(f"Optimal value is: {optimal_value}\n\n")

# %%
# Checking that there is no always a left merge, in this case optimal move is up or down but the
# move up is checked before down (get_slider_move) so assert is will pass only for this value
assert optimal_policy[(4, 2, 0, 4, 0, 0, 0, 0, 0)] == 3

# Counting non left and print some of them
# %%
count_non_left = 0
for state, action in optimal_policy.items():
    if action is not None and action != 0:
        count_non_left += 1
        if count_non_left <= 5: # Print first 5 non-left actions
            print(f"State: {np.array(state).reshape(3,3).flatten()}")
            print(f"Recommended action: {action}")
            print("----------------")

print(f"Total states where action is NOT Left: {count_non_left}")

visualize_strategic_preference(optimal_value)