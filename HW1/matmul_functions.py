import math
import numpy as np
from numba import njit, cuda
import timeit


def matmul_transpose_trivial(X):
    M = X.shape[0]
    K = X.shape[1]
    N = X.shape[0]
    Y = np.zeros((M, M))
    for m in range(M):
        for n in range(m+1):
            for k in range(K):
                Y[m, n] += X[m, k] * X[n, k]
            Y[n, m] = Y[m, n]
    return  Y
    raise NotImplementedError("To be implemented")


@njit
def matmul_transpose_numba(X):
    # return matmul_transpose_trivial(X)
    M = X.shape[0]
    K = X.shape[1]
    N = X.shape[0]
    Y = np.zeros((M, M))
    for m in range(M):
        for n in range(m+1):
            for k in range(K):
                Y[m, n] += X[m, k] * X[n, k]
            Y[n, m] = Y[m, n]
    return  Y
    raise NotImplementedError("To be implemented")


def matmul_transpose_gpu(X):
    threads_per_block = (1024)
    blocks_per_grid = (1)

    X_d = cuda.to_device(X)
    C = np.zeros((X.shape[0], X.shape[0]))
    C_d = cuda.to_device(C)
    matmul_kernel[blocks_per_grid, threads_per_block](X_d, C_d)
    C = C_d.copy_to_host()
    return C
    raise NotImplementedError("To be implemented")

@cuda.jit
def matmul_kernel(A, C):
    n_elements = int((A.shape[0] * (1 + A.shape[0])) / 2)
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    thread_section_size = int(math.floor(n_elements // bdim)) if tid < bdim - 1 else n_elements - int((bdim-1) * math.floor(n_elements // bdim))
    last_thread_section_size = n_elements - int((bdim-1) * math.floor(n_elements // bdim))
    thread_section_start = int(tid * thread_section_size) if tid < bdim -1 else int(tid * math.floor(n_elements // bdim))
    thread_section_end = int(thread_section_start + (thread_section_size if (tid < bdim-1) else last_thread_section_size))
    thread_section_size = thread_section_size if (tid < bdim-1) else last_thread_section_size
    
    for i in range(thread_section_start, thread_section_end):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        for k in range(A.shape[1]):
            C[x, y] += A[x, k] * A[y, k]
        C[y, x] = C[x, y]
    raise NotImplementedError("To be implemented")

#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()
    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X,Xt)).repeat(3, 100))

    #print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()

    X = np.reshape(np.arange(784*128), (784, 128))
    # X = np.random.randn(784, 128)
    Xt = X.copy().transpose()
    
    XXt = matmul_transpose_numba(X)
    print(not np.any(np.matmul(X, Xt) - XXt))

    XXt = matmul_transpose_gpu(X)
    print(not np.any(np.matmul(X, Xt) - XXt))
