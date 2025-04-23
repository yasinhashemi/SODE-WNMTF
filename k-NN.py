import numpy as np
from scipy.sparse import coo_array
import faiss

def KNN_graph(data, k, sigma):
    '''
    Construction of k nearest neighbor graph 
    '''

    pixel_num = data.shape[1]
    transposed_data = data.T
    metric = faiss.METRIC_L2
    index = faiss.IndexFlatL2(transposed_data.shape[1])
    index.add(transposed_data)
    
    distances, indices = index.search(transposed_data, k + 1)   # Identifying indices and distances of the k+1 closest neighbors for each pixel, including the pixel itself.
    distances, indices = np.exp((-1 / sigma) * distances[:, 1:]), indices[:, 1:]  # Converting similarity metric from Euclidean distance to heat kernel
    rows = np.tile(np.arange(pixel_num)[:, np.newaxis], k)
    W1 = (coo_array((0.5 *distances.ravel(), (rows.ravel(), indices.ravel())), shape= (pixel_num, pixel_num)) +
        coo_array((0.5 * distances.ravel(), (rows.ravel(), indices.ravel())), shape= (pixel_num, pixel_num)).T)

    return scipy.sparse.diags(W1.sum(axis= 1)), W1