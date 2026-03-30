import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    # N is the batch size, T is the sequence length and d is the input size
    N, T, d = X.shape
    # Hidden size
    m = h_0.shape[1]

    # Pre-locate h_all to store all hidden states
    h_all = np.zeros((N, T, m))

    h_prev = h_0

    for t in range(T):
        # Get the current input
        x_t = X[:, t, :]
        # Calculate the current hidden state
        h_t = np.tanh(np.matmul(h_prev, W_hh) + np.matmul(x_t, W_xh.T) + b_h)
        # Store the current state to h_all
        h_all[:, t, :] = h_t
        # Save the current state for the next step
        h_prev = h_t

    return (h_all, h_prev)