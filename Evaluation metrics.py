import numpy as np

def SAD(real_endmembers, est_endmembers):
    '''
    Computation of SAD based on the ground-truth endmembers and the estimated ones
    real_endmembers: L X K ground-truth endmember matrix
    est_endmembers: L X K estimated endmember matrix derived from GOP-WNMF
    '''
    
    sad = 0
    for i in range(real_endmembers.shape[1]):
        sad += np.arccos((real_endmembers[:, i].T @ est_endmembers[:, i]) / (norm(real_endmembers[:, i]) * norm(est_endmembers[:, i])))
    return sad / (real_endmembers.shape[1])


def RMSE(real_abundances, est_abundances):
    E = real_abundances - est_abundances
    return np.sqrt(np.mean(E * E))