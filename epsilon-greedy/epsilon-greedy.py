import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Write code here
    prob = rng.random() if rng is not None else np.random.rand()
    
    if prob < epsilon:
        return rng.integers(0, len(q_values)) if rng is not None else np.random.randint(0, len(q_values))
    else:
        return np.argmax(q_values)