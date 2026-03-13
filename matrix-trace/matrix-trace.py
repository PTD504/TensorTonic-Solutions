import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    N = len(A)
    A_trace = 0

    for i in range(N):
        A_trace += A[i][i]

    return A_trace
