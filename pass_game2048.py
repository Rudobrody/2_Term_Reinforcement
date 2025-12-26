# %%
# Import env
from game2048 import Action, Player, Adversarial2048Env
import numpy as np

# %%
# Creating env
env_2048 = Adversarial2048Env()
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

    # init with a policy with first avail action for each state
    for current_state in env.get_all_states():
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
        for state in env.get_all_states():
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
# %%
