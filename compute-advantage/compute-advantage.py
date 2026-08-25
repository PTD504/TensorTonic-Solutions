import numpy as np

def compute_advantage(states: list, rewards: list, V: list, gamma: float) -> np.ndarray:
    """
    Returns the advantages as a NumPy array.
    """
    # Compute the discounted return
    G = [0]
    for t in reversed(range(len(rewards))):
        G.append(rewards[t] + gamma * G[-1])

    G = G[::-1]
    G.pop()

    A = np.array([G[t] - V[t] for t in range(len(states))])

    return np.round(A, 4)