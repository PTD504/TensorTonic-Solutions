def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    num_rewards = len(rewards)

    if num_rewards == 1:
        return rewards

    G = [rewards[-1]]

    for i in reversed(range(len(rewards) - 1)):
        G.append(G[-1] * gamma + rewards[i])

    return G[::-1]