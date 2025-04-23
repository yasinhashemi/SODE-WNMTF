import numpy as np

def weight(X, U, S, V, mu1, mu2):
    '''
    Construction of Weighting matrix T using the co-clusternig property of NMTF
    '''

    band_centers = S @ V  # Centroids of spectral band clusters located in each row
    pixel_centers = U @ S  # Centroids of pixel clusters located in each column
    band_clusters = np.argmax(U, axis= 1)  # band cluster membership indices 
    pixel_clusters = np.argmax(V, axis= 0)  # pixel cluster membership indices

    return mu1 / norm(X - band_centers[band_clusters, :], axis= 1, keepdims= True) + mu2 / norm(X - pixel_centers[:, pixel_clusters], axis= 0, keepdims= True)