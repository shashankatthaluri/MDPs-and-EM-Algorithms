import numpy as np

# Define the states and actions
states = ['start', 'move_left', 'move_right', 'pick_up_trash', 'recycle', 'end']
actions = ['left', 'right', 'pick_up', 'recycle', 'wait']

# Define the transition probabilities for each action and state
def transition_probs(state, action):
    if state == 'start' and action == 'wait':
        return {'move_left': 1.0}
    elif state == 'start' and action == 'pick_up':
        return {'pick_up_trash': 1.0}
    elif state == 'move_left' and action == 'wait':
        return {'move_left': 1.0}
    elif state == 'move_left' and action == 'right':
        return {'start': 1.0}
    elif state == 'move_left' and action == 'pick_up':
        return {'pick_up_trash': 1.0}
    elif state == 'move_right' and action == 'wait':
        return {'move_right': 1.0}
    elif state == 'move_right' and action == 'left':
        return {'start': 1.0}
    elif state == 'move_right' and action == 'pick_up':
        return {'pick_up_trash': 1.0}
    elif state == 'pick_up_trash' and action == 'wait':
        return {'recycle': 1.0}
    elif state == 'recycle' and action == 'wait':
        return {'end': 1.0}
    else:
        return None

# Define the reward for each state
def reward(state):
    if state == 'pick_up_trash':
        return 10.0
    elif state == 'recycle':
        return 20.0
    else:
        return 0.0

# Define the discount factor
gamma = 0.9

# Define the value iteration algorithm
def value_iteration():
    # Initialize the value of each state to 0
    V = {s: 0 for s in states}
    
    # Iterate until convergence
    while True:
        # Initialize the change in value to 0
        delta = 0
        
        # Update the value of each state
        for s in states:
            v = V[s]
            
            # Calculate the maximum expected value over all actions
            max_v = -float('inf')
            for a in actions:
                tp = transition_probs(s, a)
                if tp is not None:
                    expected_v = 0
                    for s_ in tp:
                        r = reward(s_)
                        expected_v += tp[s_] * (r + gamma * V[s_])
                    max_v = max(max_v, expected_v)
            
            # Update the value of the state
            V[s] = max_v
            
            # Update the change in value
            delta = max(delta, abs(v - V[s]))
        
        # Check for convergence
        if delta < 1e-6:
            break
    
    # Return the optimal policy 
    policy = {}
    for s in states:
      policy[s] = max(actions, key=lambda a: expected_returns(s, a))

    # Print the optimal policy
    print("Optimal Policy:")
    for s in states:
      print(f"{s}: {policy[s]}")
