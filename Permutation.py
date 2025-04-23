import numpy as np

def Permutation(real_endmembers, real_abundances, est_endmembers, est_abundances):
    '''
    Permutation of the estimated endmember and abundance matrices using the correlation coefficient metric with respect to their corresponding ground truth.
    Each estimated endmember signature is matched to the column in the ground-truth endmember matrix with which it has the highest correlation.
    '''
    
    endmembers_num = real_endmembers.shape[1]
    final_endmembers = np.zeros_like(est_endmembers)
    final_abundances = np.zeros_like(est_abundances)

    for i in range(endmembers_num):
        corr_arr = np.zeros(endmembers_num)
        for j in range(endmembers_num):
            corr_arr[j] = np.corrcoef(est_abundances[i, :], real_abundances[j, :])[0, 1]  # Computing the correlation coefficients between the rows of the estimated
            # and ground-truth abundance matrices.
        ind = np.argmax(corr_arr)
        final_endmembers[:, ind] = est_endmembers[:, i]
        final_abundances[ind, :] = est_abundances[i, :]
        

    return final_endmembers, final_abundances