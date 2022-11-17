import os
import numpy as np
from numba import cuda, njit, prange, float32
import timeit


def max_cpu(A, B):
    """
    Returns
    -------
    np.array
        element-wise maximum between A and B
    """
    # return np.maximum(A, B)
    C = np.empty_like(A)
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            a = A[x, y]
            b = B[x, y]
            C[x, y] = max(a, b)
    return C
    pass

# NumbaPerformanceWarning: The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible..
# @njit(parallel=True)
@njit()
def max_numba(A, B):
    """
    Returns
    -------
    np.array
        element-wise maximum between A and B
    """
    # return np.maximum(A, B)
    C = np.empty_like(A)
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            C[x, y] = max(A[x, y], B[x, y])
    return C
    pass


@cuda.jit
def max_gpu_single_thread(A, B, C):
    x, y = cuda.grid(2)
    if x < A.shape[0] and y < A.shape[1]:
        C[x, y] = max(A[x, y], B[x,y])

def max_gpu(A, B):
    """
    Returns
    -------
    np.array
        element-wise maximum between A and B
    """

    threads_per_block = (1, A.shape[1])
    blocks_per_grid = (A.shape[0], 1)


    A_d = cuda.to_device(A)
    B_d = cuda.to_device(B)
    C = np.empty_like(A)
    C_d = cuda.to_device(C)
    
    max_gpu_single_thread[blocks_per_grid, threads_per_block](A_d, B_d, C_d)

    C = C_d.copy_to_host()

    return C


@cuda.jit
def max_kernel(A, B, C):
    pass


# this is the comparison function - keep it as it is.
def max_comparison():
    N = 1000
    A = np.random.randint(0, 256, (N, N))
    B = np.random.randint(0, 256, (N, N))

    def timer(f):
        return min(timeit.Timer(lambda: f(A, B)).repeat(3, 20))

    print('     [*] CPU:', timer(max_cpu))
    print('     [*] Numba:', timer(max_numba))
    print('     [*] CUDA:', timer(max_gpu))


if __name__ == '__main__':
    os.system('lscpu')
    os.system('nvidia-smi')
    os.system('nvidia-smi --query-gpu=name --format=csv')
    max_comparison()

    # N = 1000
    # A = np.random.randint(0, 256, (N, N))
    # B = np.random.randint(0, 256, (N, N))

    # print('max_cpu')
    # print(not np.any(np.maximum(A, B) - max_cpu(A, B)))

    # print('max_numba')
    # print(not np.any(np.maximum(A, B) - max_numba(A, B)))

    # print('max_gpu')
    # print(not np.any(np.maximum(A, B) - max_gpu(A, B)))