def Tri_NMF(X, V, endmembers_VCA, real_endmembers, real_abundances, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, sigma_V, k_V, U_cluster, mu1, mu2, p= 0.8, delta= 15, Max_Iter= 3000, error_tol= 1e-7):
    '''
    An implementation of Simultaneous Outlier Detection and Elimination in Hyperspectral Unmixing via Weighted Non-negative Matrix Tri-Factorization
    Seyedyasin Hashemi Nazari
    Email: yasinhashemi13@gmail.com
    23-4-2025

    SODE-WNMTF: Proposed Method

    Inputs
    X: Hyperspectral image in R_+^(L X N), where L and N are the numbers of spectral bands and pixels, respectively.
    V: Abundance matrix in R_+^(K X N), where K is the number of endmembers.
    endmembers_VCA: Estimated endmembers in R_+^(L X K) via VCA method.
    real_endmembers: Ground-truth endmembers in R_+^(L X K).
    alpha_1, ..., alpha_6: Regularization hyper parameters
    sigma_V: Band width utilized in heat kernel
    k_V: Number of neighbors for K-NN graph construction
    U_cluster: Number of spectral band clusters
    mu1, mu2: Hyper parameters associated with weighting matrix T
    p: Coefficient of the augmented row in Tbar
    delta: Coefficient of the augmented row in Xbar and Mbar
    Max_Iter: Maximum number of iterations
    error_tol: Error tolerance
    '''
    
    cnt= 0
    D_V, W_V = KNN_graph(X, k_V, sigma_V)
    np.random.seed(101)
    # Initialization
    U = np.abs(np.random.random((X.shape[0], U_cluster)))
    S = np.abs(np.linalg.pinv(U) @ endmembers_VCA)
    W = np.random.rand(X.shape[1], S.shape[1])
    T = weight(X, U, S, V, mu1, mu2)
    A = sparsity(X, U, S, V, mu1, mu2)

    former_residual = RMSE(X, U @ S @ V)

    for iter in range(Max_Iter):
        # Update U
        numU = (T * T * X) @ V.T @ S.T + alpha_1 * (X @ W @ S.T) + alpha_3 * U
        denU = (T * T * (U @ S @ V)) @ V.T @ S.T + alpha_1 * (U @ S @ S.T) + alpha_3 * (U @ U.T @ U)
        reU = numU / np.maximum(denU, 1e-10)
        U = U * reU
        U[U < 1e-4] = 1e-4

        # Update S
        numS = U.T @ (T * T * X) @ V.T + alpha_1 * (U.T @ X @ W)
        denS = U.T @ (T * T * (U @ S @ V)) @ V.T + alpha_1 * (U.T @ U @ S)
        reS = numS / np.maximum(denS, 1e-10)
        S = S * reS
        S[S < 1e-4] = 1e-4

        # Update V
        Xbar = np.concatenate([X, delta * np.ones((1, X.shape[1]))])
        M = U @ S
        Mbar = np.concatenate([M, delta * np.ones((1, M.shape[1]))])
        V[V <= 1e-4] = 1e-4
        Tbar = np.concatenate([T, p * np.ones((1, T.shape[1]))])

        numV = Mbar.T @ (Tbar * Tbar * Xbar) + alpha_5 * (V @ W_V)
        denV = Mbar.T @ (Tbar * Tbar * (Mbar @ V)) + alpha_5 * (V @ D_V) + alpha_4 * A
        reV = numV / np.maximum(denV, 1e-10)
        V = V * reV

        # Update W
        G = X @ W
        Q = X.T @ G
        I = X @ Q
        numW = alpha_1 * (X.T @ U @ S) + alpha_2 * (X.T @ I)
        denW = alpha_1 * Q + alpha_2 * (Q @ G.T @ G) + alpha_6 * (W ** (-0.5))
        reW = numW / np.maximum(denW, 1e-10)
        W = W * reW
        W[W < 1e-4] = 1e-4

        # Update A
        A = sparsity(X, U, S, V, mu1, mu2)

        # Update T
        T = weight(X, U, S, V, mu1, mu2)

        latter_residual = RMSE(X, U @ S @ V)
        err = np.abs(former_residual - latter_residual)
        former_residual = latter_residual

        if err < error_tol:
            cnt += 1
            if cnt == 10:
                break
    
    E, R = Permutation(real_endmembers, real_abundances, U @ S, V)

    sad = SAD(real_endmembers, E)
    rmse = RMSE(real_abundances, R)

    return sad, rmse, cnt, iter, E, R, A
