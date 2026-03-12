import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # convert to numpy array
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)

    # Update the running average of the gradient (momentum)
    m_t = beta1 * m + (1 - beta1) * grad

    # Update the running average of the squared gradient (RMSprop feature)
    v_t = beta2 * v + (1 - beta2) * (grad ** 2)

    # Calculate the corrected m and v (bias correction)
    m = m_t / (1 - (beta1 ** t))
    v = v_t / (1 - (beta2 ** t))

    # Update the next step (Adam step)
    param_t = param - lr * m / (np.sqrt(v) + eps)

    return (param_t, m_t, v_t)