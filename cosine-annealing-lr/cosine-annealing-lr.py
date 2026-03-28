def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    lr = 1 + math.cos(math.pi * current_step / total_steps)
    lr *= 0.5 * (base_lr - min_lr)
    lr += min_lr

    return lr