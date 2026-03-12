import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # convert to numpy array
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)

    # Update the running average of the squared gradient
    s = beta * s + (1 - beta) * (g ** 2)

    # Update the next step
    w = w - lr * g / np.sqrt(s + eps)

    return (w, s)