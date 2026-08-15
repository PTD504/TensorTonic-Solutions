import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    # Calculate the target
    target = r + gamma * V[s_next]

    # Calculate the TD error
    error = target - V[s]

    # Update step
    V[s] += alpha * error

    return V