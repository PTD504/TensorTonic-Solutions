import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Compute the dot product between a and b
    dot_product = np.dot(a, b)
    
    # Compute the product of magnitudes
    a_product_of_magnitudes = np.linalg.norm(a)
    b_product_of_magnitudes = np.linalg.norm(b)

    if a_product_of_magnitudes == 0 or b_product_of_magnitudes == 0:
        return 0

    return dot_product / (a_product_of_magnitudes * b_product_of_magnitudes)