import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    try:
        dot_product = np.sum(np.array(x) * np.array(y))
    except ValueError as e:
        raise

    return dot_product