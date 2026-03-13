import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    n = len(v)
    diagonal_matrix = np.zeros((n, n))

    for i in range(n):
        diagonal_matrix[i][i] = v[i]

    return diagonal_matrix
