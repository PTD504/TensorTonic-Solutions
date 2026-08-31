def gae(rewards: list, values: list, gamma: float, lam: float) -> list:
    """
    Returns the generalized advantage estimate at every timestep.
    """
    delta = []

    for t in range(len(rewards)):
        delta.append(rewards[t] + gamma * values[t + 1] - values[t])

    adv = [delta[-1]]

    for t in reversed(range(len(rewards) - 1)):
        adv.append(delta[t] + gamma * lam * adv[-1])

    return adv[::-1]