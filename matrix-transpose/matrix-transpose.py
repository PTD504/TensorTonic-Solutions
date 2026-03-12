import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    N, M = len(A), len(A[0])
    A_T = []

    for j in range(0, M):
        A_T_row = []
        for i in range(0, N):
            A_T_row.append(A[i][j])
        A_T.append(A_T_row)

    return np.array(A_T)