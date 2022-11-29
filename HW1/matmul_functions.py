import math
import numpy as np
import numba
from numba import njit, cuda
import timeit


M = 784
K = 128
N_ELEMENTS = int((M * (1 + M)) / 2)
SHMEM_M = M#math.floor(49152 / 8 / N)  # 2080TI has 48[KB] of shared memory. Plus, we assume below float64 for A.
SHMEM_N = math.floor(49152 / 8 / M)#N
SHMEM_ELEMENTS = SHMEM_M * SHMEM_N  
TPB = (1024)#(32,32)  # Threads per Block
BPG = (1)#(1,1)  # Blocks per Grid
# SECTION = (math.ceil(M/TPB[0]), math.ceil(M/TPB[1]))
#CPT = N_ELEMENTS // TPB  # Cells per Thread
#CPT_PLUS_ONE = N_ELEMENTS % TPB  # Threads [0,CPT_PLUS_ONE-1] gets CPT+1 cells each
PREFETCH_WINDOW = 7

SHMEM_SIZE = 49152


def matmul_transpose_trivial(X):
    M = X.shape[0]
    K = X.shape[1]
    Y = np.zeros((M, M))
    for m in range(M):
        for n in range(m+1):
            # for k in range(K):
            for k in range(K):
                Y[m, n] += X[m, k] * X[n, k]
            Y[n, m] = Y[m, n]
    return  Y


@njit
def matmul_transpose_numba(X):
    # return matmul_transpose_trivial(X)
    M = X.shape[0]
    K = X.shape[1]
    Y = np.zeros((M, M))
    for m in range(M):
        for n in range(m+1):
            for k in range(K):
                Y[m, n] += X[m, k] * X[n, k]
            Y[n, m] = Y[m, n]
    return  Y


def matmul_transpose_gpu(X):
   
    X_d = cuda.to_device(X)
    C_d = cuda.device_array((M, M))

    matmul_kernel[BPG, TPB](X_d, C_d)
    
    C = C_d.copy_to_host()

    return C

def matmul_transpose_gpu_interleaving_threads_inner_prod(X):
   
    X_d = cuda.to_device(X)
    C_d = cuda.device_array((M, M))

    matmul_kernel_interleaving_threads_inner_prod[BPG, TPB](X_d, C_d)
    
    C = C_d.copy_to_host()

    return C

def matmul_transpose_gpu_interleaving_threads_outer_prod(X):
   
    X_d = cuda.to_device(X)
    C_d = cuda.device_array((M, M))

    matmul_kernel_interleaving_threads_outer_prod[BPG, TPB](X_d, C_d)
    
    C = C_d.copy_to_host()

    return C

def matmul_transpose_gpu_interleaving_threads_inner_prod_with_vannila_shmem(X):
   
    X_d = cuda.to_device(X)
    C_d = cuda.device_array((M, M))

    matmul_kernel_interleaving_threads_inner_prod_with_vannila_shmem[BPG, TPB](X_d, C_d)
    
    C = C_d.copy_to_host()

    return C

def matmul_transpose_gpu_inner_prod(X):
   
    X_d = cuda.to_device(X)
    C_d = cuda.device_array((M, M))

    matmul_kernel_base_inner_prod[BPG, TPB](X_d, C_d)
    
    C = C_d.copy_to_host()

    return C

def matmul_transpose_gpu_outer_prod(X):
   
    X_d = cuda.to_device(X)
    C_d = cuda.device_array((M, M))

    matmul_kernel_base_outer_prod[BPG, TPB](X_d, C_d)
    
    C = C_d.copy_to_host()

    return C

# Work distribution to threads interleaving fashion:
# T0->C[0,0], T1->C[1,0], T2->C[1,1], T3->C[2,0], ..., T1023->C[44,33], T0->C[44,34], ...
# Still no shared memory
# Inner-Product fashion
# 3.6[s]
@cuda.jit
def matmul_kernel_interleaving_threads_inner_prod(A, C):
    
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    m, n = C.shape[0], C.shape[1]
    k = A.shape[1]
    n_elements = int(m * (m + 1) / 2)

    for i in range(tid, n_elements, bdim):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        for kk in range(k):
            C[x, y] += A[x, kk] * A[y, kk]
        C[y, x] = C[x, y]


# Work distribution to threads interleaving fashion:
# T0->C[0,0], T1->C[1,0], T2->C[1,1], T3->C[2,0], ..., T1023->C[44,33], T0->C[44,34], ...
# Still no shared memory
# Outer-Product fashion
# 20.6[s]
@cuda.jit
def matmul_kernel_interleaving_threads_outer_prod(A, C):
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    m, n = C.shape[0], C.shape[1]
    k = A.shape[1]
    n_elements = int(m * (m + 1) / 2)

    for kk in range(k):
        for i in range(tid, n_elements, bdim):
            x = int((math.sqrt(8 * i + 1) - 1) / 2)
            y = int(i - x * (x + 1) / 2)
            # for k in range(N):
            C[x, y] += A[x, kk] * A[y, kk]
            # C[y, x] = C[x, y]
    for i in range(tid, n_elements, bdim):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        C[y, x] = C[x, y]


# Work distribution to threads interleaving fashion:
# T0->C[0,0], T1->C[1,0], T2->C[1,1], T3->C[2,0], ..., T1023->C[44,33], T0->C[44,34], ...
# With shared memory only for A rows [0,47) - this is as much as the shared memory can contain
# Inner-Product fashion
# 3.8[s]
@cuda.jit
def matmul_kernel_interleaving_threads_inner_prod_with_vannila_shmem(A, C):
    
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    m = C.shape[0]
    k = A.shape[1]
    n = C.shape[1]
    n_elements = int(m * (m + 1) / 2)
    shmem_m = m
    shmem_k = int(SHMEM_SIZE / 8 / shmem_m)
    shmem_elements = shmem_m * shmem_k

    sA = cuda.shared.array(shape=(784, 7), dtype=numba.float64)  # TODO: Using shmem_m,shmem_k here is not working. why?

    for i in range(tid, shmem_elements, bdim):
        y = int(i % shmem_k)
        x = int(i / shmem_k)
        sA[x,y] = A[x,y]
    
    cuda.syncthreads()

    for i in range(tid, n_elements, bdim):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        
        for kk in range(k):
            if kk < shmem_k:
                C[x, y] += sA[x, kk] * sA[y, kk]
            else:
                C[x, y] += A[x, kk] * A[y, kk]
            # C[x, y] += A[x, kk] * A[y, kk]
        C[y, x] = C[x, y]


# Work distribution to threads interleaving fashion:
# T0->C[0,0], T1->C[1,0], T2->C[1,1], T3->C[2,0], ..., T1023->C[44,33], T0->C[44,34], ...
# With shared memory. Prefetching A rows in chunks
# Currently not working!
# Inner-Product fashion
# 
@cuda.jit
def matmul_kernel_interleaving_threads_inner_prod_with_mocca_shmem(A, C):
    
    raise NotImplemented("Not Working!!!. The prefetching is ok, but then O can't make the compute part to work only on shared memory A rows...")
    
    tid = cuda.threadIdx.x

    sA = cuda.shared.array(shape=(SHMEM_M, SHMEM_N), dtype=numba.float64)

    for m in range(0, M, SHMEM_M):
        
        # Prefetching A block of size 48 rows x 128 cols
        for i in range(tid, SHMEM_ELEMENTS, TPB):
            y = int(i % N)
            x = int(i / N)
            # if tid == 1023:
            #     print('m', m, 'x', x+m, 'y', y)
            sA[x,y] = A[x+m,y]
    
        cuda.syncthreads()

        for i in range(tid, N_ELEMENTS, TPB):
            x = int((math.sqrt(8 * i + 1) - 1) / 2)
            y = int(i - x * (x + 1) / 2)
            
            if x < SHMEM_M:
                for k in range(N):
                    C[x, y] += sA[x, k] * sA[y, k]
                C[y, x] = C[x, y]
            elif y < SHMEM_M:
                for k in range(N):
                    C[x, y] += A[x, k] * sA[y, k]
                C[y, x] = C[x, y]
            else:
                for k in range(N):
                    C[x, y] += A[x, k] * A[y, k]
                C[y, x] = C[x, y]


@cuda.jit
def matmul_kernel_base_inner_prod(A, C):

    tidx, tidy = cuda.threadIdx.x, cuda.threadIdx.y
    bdimx, bdimy = cuda.blockDim.x, cuda.blockDim.y
    M, N = C.shape
    K = A.shape[1]
    secx, secy = math.ceil(M/bdimx), math.ceil(N/bdimy)
    
    for x in range(tidx * secx, (tidx + 1) * secx):
        for y in range(tidy * secy, (tidy + 1) * secy):
            if x < M and y <= x:
                for k in range(K):
                    C[x, y] += A[x, k] * A[y, k]
                C[y, x] = C[x, y]


@cuda.jit
def matmul_kernel_base_outer_prod(A, C):
    
    tidx, tidy = cuda.threadIdx.x, cuda.threadIdx.y
    bdimx, bdimy = cuda.blockDim.x, cuda.blockDim.y
    M, N = C.shape
    K = A.shape[1]
    secx, secy = math.ceil(M/bdimx), math.ceil(N/bdimy)
    
    for k in range(K):
        for x in range(tidx * secx, (tidx + 1) * secx):
            for y in range(tidy * secy, (tidy + 1) * secy):
                if x < M and y <= x:
                    C[x, y] += A[x, k] * A[y, k]
    for x in range(tidx * secx, (tidx + 1) * secx):
        for y in range(tidy * secy, (tidy + 1) * secy):
            if x < M and y <= x:
                C[y, x] = C[x, y]


@cuda.jit
def matmul_kernel_base_outer_prod_with_shmem(A, C):
    raise NotImplemented('Still not working')
    tidx, tidy = cuda.threadIdx.x, cuda.threadIdx.y
    bdimx, bdimy = cuda.blockDim.x, cuda.blockDim.y
    tidg = tidx * bdimy + tidy  # Thread ID global
    bdimg = bdimx * bdimy  # Threads count
    M, N = C.shape
    K = A.shape[1]
    secx, secy = math.ceil(M/bdimx), math.ceil(N/bdimy)
    
    sA = cuda.shared.array(shape=(784, 4), dtype=numba.float64)

    # Transform space 32x32 to 256x4
    tidx_ = int((tidx * bdimy + tidy) / 4)
    tidy_ = (tidx * bdimy + tidy) % 4

    for k in range(0, K, 4):
        for m in range(0, M, 256):
            if k in [0,4,8] and tidx_ in [255] and tidy_ in [3]:
                print(tidx_+m, tidy_+k, '->', tidx_+m, tidy_)
            # sA[tidx_+m, tidy_] = A[tidx_+m,tidy_+k]
    
        cuda.syncthreads()

        for x in range(tidx * SECTION[0], (tidx + 1) * SECTION[0]):
            for y in range(tidy * SECTION[1], (tidy + 1) * SECTION[1]):
                if x < M and y <= x:
                    C[x, y] += A[x, k] * A[y, k]
                
    for x in range(tidx * SECTION[0], (tidx + 1) * SECTION[0]):
            for y in range(tidy * SECTION[1], (tidy + 1) * SECTION[1]):
                if x < M and y <= x:
                    C[y, x] = C[x, y]
            

#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()
    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X,Xt)).repeat(3, 100))

    #print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA (Interleaving Threads, Inner Product):', timer(matmul_transpose_gpu_interleaving_threads_inner_prod, 1))
    print('CUDA (Interleaving Threads, Outer Product):', timer(matmul_transpose_gpu_interleaving_threads_outer_prod, 1))
    print('CUDA (Interleaving Threads, Inner Product, With Vannila Shared Memory):', timer(matmul_transpose_gpu_interleaving_threads_inner_prod_with_vannila_shmem, 1))
    print('CUDA (Blocked Thread, Inner Product):', timer(matmul_transpose_gpu_inner_prod, 1))
    print('CUDA (Blocked Thread, Outer Product):', timer(matmul_transpose_gpu_outer_prod, 1))


if __name__ == '__main__':
    matmul_comparison()

    # X = np.reshape(np.arange(M*K), (M, K))
    # Xt = X.copy().transpose()
    
    # XXt_baseline = np.matmul(X, Xt)
    # XXt = matmul_transpose_numba(X)
    # print(not np.any(XXt_baseline - XXt))

    # XXt = matmul_transpose_gpu(X)
    # print(not np.any(XXt_baseline - XXt))
    
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)