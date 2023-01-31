import numpy as np

def custom_diag(X, trans=False):
    """
    Average and variance over the diagonals
    X (timesteps, dim1, dim2)
    if trans = True - compute lower and upper diagonals
    else: assume that lower triangle is zero
    """
    assert X.shape[1] == X.shape[2]
    avg_distance = []
    std_distance = []
    for i in range(X.shape[0]):
        if trans:
            diag = np.vstack((np.diagonal(X, i, axis1=1, axis2=2),np.diagonal(X, -i, axis1=1, axis2=2))).reshape(X.shape[0],-1)
        else:
            diag = np.diagonal(X, i, axis1=1, axis2=2).reshape(X.shape[0],-1)
        avg_distance.append(diag.mean(axis=1))
        std_distance.append(diag.std(axis=1))
    
    return np.array(avg_distance), np.array(std_distance)
