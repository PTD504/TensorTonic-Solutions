def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    """
    Returns sampling probabilities and normalized importance weights.
    """
    powered_priorities = [p ** alpha for p in priorities]
    sum_powered_priorities = sum(powered_priorities)

    probabilities = [powered_p / sum_powered_priorities for powered_p in powered_priorities]

    N = len(priorities)
    weights = [(N * prob) ** (-beta) for prob in probabilities]

    max_weight = max(weights)
    normalized_weights = [weight / max_weight for weight in weights]

    return [probabilities, normalized_weights]