import numpy as np
from scipy.signal import convolve2d

def sparsity(X, U, S, V, mu1, mu2):
    '''
    Construction of weighting matrix A, which enforces element-wise sparsity in the abundance matrix.
    '''

    outlier_weight = weight(X, U, S, V, mu1, mu2)  # computing tau for pixels
    # element-wise multiplication of estimated abundance matrix entries with their corresponding outlier weights (tau)
    sum_weight = np.ones_like(V) * norm(outlier_weight, axis= 0)
    weight_matrix = V * norm(outlier_weight, axis= 0)

    row = col = np.sqrt(V.shape[1]).astype(np.int_)
    endmembers_num = V.shape[0]

    tensor_weight = np.reshape(weight_matrix.copy().T, (row, col, endmembers_num))
    tensor_sum = np.reshape(sum_weight.copy().T, (row, col, endmembers_num))

    A = np.zeros_like(tensor_weight)
    kernel = np.ones((3, 3))

    # Convolution operation 
    for i in range(endmembers_num):
        A[:, :, i] = convolve2d(tensor_sum[:, :, i], kernel, mode= 'same', boundary= 'symm') / (convolve2d(tensor_weight[:, :, i], kernel, mode= 'same', boundary= 'symm') + 1e-8)

    
    unfold_A = np.reshape(A, (row * col, endmembers_num)).T
    return unfold_A