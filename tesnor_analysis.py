import numpy as np

def custom_diag(X, trans=False):
    """
    Average and variance over the diagonals
    
    if trans = True - compute lower and upper diagonals
    else: assume that lower triangle is zero
    """
    assert X.shape[0] != X.shape[1]
    avg_distance = []
    std_distance = []
    for i in range(X.shape[0]):
        if trans:
            diag = np.diagonal(X, i).reshape(-1)
        avg_distance.append(diag.mean())
        std_distance.append(diag.std())
    
    return np.array(avg_distance), np.array(std_distance)
