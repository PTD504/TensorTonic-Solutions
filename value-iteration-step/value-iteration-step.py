def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    n_states = len(values)
    # Create a new list to store results for synchronous update
    new_values = [0.0] * n_states

    for s in range(n_states):
        # Initialize best value for current state
        best_v = float('-inf')
        
        # Iterate through all available actions in state s
        for a in range(len(transitions[s])):
            # Use a loop to calculate the expected value across next states
            expected_v = 0.0
            for s_next in range(n_states):
                prob = transitions[s][a][s_next]
                expected_v += prob * values[s_next]
            
            # Bellman equation: Q(s, a) = R(s, a) + gamma * expected_v
            q_value = rewards[s][a] + gamma * expected_v
            
            if q_value > best_v:
                best_v = q_value
                
        new_values[s] = best_v
        
    return new_values