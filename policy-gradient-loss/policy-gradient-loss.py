def policy_gradient_loss(log_probs: list, rewards: list, gamma: float) -> float:
    """
    Returns the REINFORCE loss with a mean-return baseline.
    """
    # Compute the discounted returns
    discounted_return = 0.0
    G = []

    for t in reversed(range(len(rewards))):
        discounted_return = gamma * discounted_return + rewards[t]
        G.append(discounted_return)
    G = G[::-1]

    # Compute the mean return
    G_avg = sum(G) / len(rewards)

    L = 0.0

    # Compute the loss
    for t in range(len(log_probs)):
        L += log_probs[t] * (G[t] - G_avg)

    return -L / len(log_probs)