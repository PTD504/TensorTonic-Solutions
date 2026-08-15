import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    V = np.zeros(n_states)
    N = np.zeros(n_states)

    for episode in episodes:
        first_visits = {}

        for t, (s, r) in enumerate(episode):
            if s not in first_visits:
                first_visits[s] = t

        G = 0.0

        for step in reversed(range(len(episode))):
            s, r = episode[step]
            G = r + gamma * G

            if step == first_visits[s]:
                N[s] += 1
                V[s] += (G - V[s]) / N[s]

    return V
