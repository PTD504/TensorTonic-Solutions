import numpy as np

def replay_buffer_sample(buffer: list, batch_size: int, seed: int) -> list:
    """
    Returns a deterministic sample of transitions.
    """
    np.random.seed(seed)

    replay_indices = np.random.choice(len(buffer), size=batch_size, replace=False)

    sorted_replay_indices = np.sort(replay_indices)

    return [buffer[idx] for idx in sorted_replay_indices]