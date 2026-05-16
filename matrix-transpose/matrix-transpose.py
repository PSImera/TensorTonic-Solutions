import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # return np.array(A).T

    np_A = np.array(A)
    n, m = np_A.shape[0], np_A.shape[1]

    result = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            result[i][j] = np_A[j][i]

    return result