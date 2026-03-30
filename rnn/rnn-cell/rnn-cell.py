import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    # YOUR CODE HERE
    # x_t: (1, d)
    # h_prev: (1, m)
    # W_xh - Weight matrix for input-to-hidden connection: (m, d)
    # W_hh - Weight matrix for hidden-to-hidden connection: (m, m)
    # b_h - Bias vector for hidden state: (m,)
    lin_pre_hidden = np.matmul(h_prev, W_hh)
    
    lin_cur_input = np.matmul(x_t, W_xh.T)

    h_t_before_tanh = lin_pre_hidden + lin_cur_input + b_h

    h_t = np.tanh(h_t_before_tanh)
    
    return h_t