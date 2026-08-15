import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    # Ensure Q is a float64 NumPy array to prevent int64 dtype constraints
    Q_new = np.array(Q, dtype=np.float64, copy=True)

    # Compute the TD Error - The difference between the TD Target (sum of the immediate reward and the maximum Q value of the next state) and the old estimated value of Q function in the current state
    td_error = r + gamma * np.max(Q_new[s_next]) - Q_new[s][a]

    # Update step
    Q_new[s][a] += alpha * td_error

    return Q_new