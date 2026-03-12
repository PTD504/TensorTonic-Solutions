def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """

    while steps > 0:
        # Compute the gradient of the quadratic at x0
        gradient = 2 * a * x0 + b

        # Compute the value of x in the next step
        x0 = x0 - lr * gradient

        # Decrement the steps by 1 each loop iteration
        steps -= 1

    return x0
        