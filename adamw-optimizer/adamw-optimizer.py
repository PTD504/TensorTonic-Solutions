import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Convert to numpy array
    w = np.array(w)
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)
    
    # Update first moment - calculate m_t using m_{t-1} and gradient at the current step
    m = beta1 * m + (1 - beta1) * grad

    # Update second moment - calculate v_t using v_{t-1} and the squared gradient at the current step
    v = beta2 * v + (1 - beta2) * grad ** 2

    # AdamW Parameter Update
    w = w - lr * (weight_decay * w) - lr * m / (np.sqrt(v) + eps)

    return (w, m, v)
    