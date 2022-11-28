import math
import numpy as np
import numba
from numba import njit, cuda
import timeit


M = 784
N = 128
N_ELEMENTS = int((M * (1 + M)) / 2)
TPB = (1024)  # Threads per Block
BPG = (1)  # Blocks per Grid
CPT = N_ELEMENTS // TPB  # Cells per Thread
CPT_PLUS_ONE = N_ELEMENTS % TPB  # Threads [0,CPT_PLUS_ONE-1] gets CPT+1 cells each
PREFETCH_WINDOW = 7

def matmul_transpose_trivial(X):
    # M = X.shape[0]
    # K = X.shape[1]
    Y = np.zeros((M, M))
    for m in range(M):
        for n in range(m+1):
            # for k in range(K):
            for k in range(N):
                Y[m, n] += X[m, k] * X[n, k]
            Y[n, m] = Y[m, n]
    return  Y


@njit
def matmul_transpose_numba(X):
    # return matmul_transpose_trivial(X)
    # M = X.shape[0]
    # K = X.shape[1]
    Y = np.zeros((M, M))
    for m in range(M):
        for n in range(m+1):
            # for k in range(K):
            for k in range(N):
                Y[m, n] += X[m, k] * X[n, k]
            Y[n, m] = Y[m, n]
    return  Y


def matmul_transpose_gpu(X):
   
    X_d = cuda.to_device(X)
    C_d = cuda.device_array((M, M))

    # matmul_kernel[BPG, TPB](X_d, C_d)
    # matmul_kernel_improved_load_balance_inner_prod[BPG, TPB](X_d, C_d)
    # matmul_kernel_improved_load_balance_outer_prod[BPG, TPB](X_d, C_d)

    # matmul_kernel_vanilla_shmem_inner_prod[BPG, TPB](X_d, C_d)
    # matmul_kernel_vanilla_shmem_outer_prod[BPG, TPB](X_d, C_d)
    
    # matmul_kernel_interleaving_threads_inner_prod[BPG, TPB](X_d, C_d)
    matmul_kernel_interleaving_threads_outer_prod[BPG, TPB](X_d, C_d)

    C = C_d.copy_to_host()

    return C

# Basic CUDA Matrix Mult X @ X'. 
# Unballanced load balancing: 1023 threads x 300 tasks, 1 thread x 820 tasks.
# No shared memory.
# ~8.8[s]
@cuda.jit
def matmul_kernel_base(A, C):

    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x

    thread_section_size = int(math.floor(N_ELEMENTS // bdim)) if tid < bdim - 1 else N_ELEMENTS - int((bdim-1) * math.floor(N_ELEMENTS // bdim))
    last_thread_section_size = N_ELEMENTS - int((bdim-1) * math.floor(N_ELEMENTS // bdim))
    thread_section_start = int(tid * thread_section_size) if tid < bdim -1 else int(tid * math.floor(N_ELEMENTS // bdim))
    thread_section_end = int(thread_section_start + (thread_section_size if (tid < bdim-1) else last_thread_section_size))
    thread_section_size = thread_section_size if (tid < bdim-1) else last_thread_section_size

    for i in range(thread_section_start, thread_section_end):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        # for k in range(A.shape[1]):
        for k in range(N):
            C[x, y] += A[x, k] * A[y, k]
        C[y, x] = C[x, y]

# Improved load balancing: 520 threads x 301 tasks, 504 threads x 300 tasks.
# Still no shared memory
# Inner-Product fashion
# ~8.1[s]
@cuda.jit
def matmul_kernel_improved_load_balance_inner_prod(A, C):
    tid = cuda.threadIdx.x
    if tid < CPT_PLUS_ONE:
        cpt = CPT + 1
        section_start = tid * cpt
    else:
        cpt = CPT
        section_start = CPT_PLUS_ONE * (cpt + 1) + (tid - CPT_PLUS_ONE) * cpt
    section_end = section_start + cpt - 1
    
    for i in range(section_start, section_end+1):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        for k in range(N):
            C[x, y] += A[x, k] * A[y, k]
        C[y, x] = C[x, y]


# Improved load balancing: 520 threads x 301 tasks, 504 threads x 300 tasks.
# Still no shared memory
# Outer-Product fashion
# ~20.8[s]
@cuda.jit
def matmul_kernel_improved_load_balance_outer_prod(A, C):
    tid = cuda.threadIdx.x
    if tid < CPT_PLUS_ONE:
        cpt = CPT + 1
        section_start = tid * cpt
    else:
        cpt = CPT
        section_start = CPT_PLUS_ONE * (cpt + 1) + (tid - CPT_PLUS_ONE) * cpt
    section_end = section_start + cpt - 1
    
    for k in range(N):
        for i in range(section_start, section_end+1):
            x = int((math.sqrt(8 * i + 1) - 1) / 2)
            y = int(i - x * (x + 1) / 2)
            # for k in range(N):
            C[x, y] += A[x, k] * A[y, k]
            # C[y, x] = C[x, y]
    for i in range(section_start, section_end+1):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        # for k in range(N):
        C[y, x] = C[x, y]


# Word distribution to threads interleaving fashion:
# T0->C[0,0], T1->C[1,0], T2->C[1,1], T3->C[2,0], ..., T1023->C[44,33], T0->C[44,34], ...
# Still no shared memory
# Inner-Product fashion
# 3.6[s]
@cuda.jit
def matmul_kernel_interleaving_threads_inner_prod(A, C):
    tid = cuda.threadIdx.x
    for i in range(tid, N_ELEMENTS, TPB):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        for k in range(N):
            C[x, y] += A[x, k] * A[y, k]
        C[y, x] = C[x, y]


# Word distribution to threads interleaving fashion:
# T0->C[0,0], T1->C[1,0], T2->C[1,1], T3->C[2,0], ..., T1023->C[44,33], T0->C[44,34], ...
# Still no shared memory
# Outer-Product fashion
# 20.6[s]
@cuda.jit
def matmul_kernel_interleaving_threads_outer_prod(A, C):
    tid = cuda.threadIdx.x
    for k in range(N):
        for i in range(tid, N_ELEMENTS, TPB):
            x = int((math.sqrt(8 * i + 1) - 1) / 2)
            y = int(i - x * (x + 1) / 2)
            # for k in range(N):
            C[x, y] += A[x, k] * A[y, k]
            # C[y, x] = C[x, y]
    for i in range(tid, N_ELEMENTS, TPB):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        C[y, x] = C[x, y]


# Word distribution to threads interleaving fashion:
# T0->C[0,0], T1->C[1,0], T2->C[1,1], T3->C[2,0], ..., T1023->C[44,33], T0->C[44,34], ...
# With shared memory
# Inner-Product fashion
# 
@cuda.jit
def matmul_kernel_interleaving_threads_inner_prod_with_shmem(A, C):
    raise NotImplemented  # need to add the shmem perfetch loop
    tid = cuda.threadIdx.x
    for i in range(tid, N_ELEMENTS, TPB):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        for k in range(N):
            C[x, y] += A[x, k] * A[y, k]
        C[y, x] = C[x, y]


# Improved load balancing + shared memory only on cols 0-6 of A
# Shared Memory in 2080TI is 48[KB]. Assuming 8[B] floats64, we can store floor(48*1024/(784*8)) cols of A at the same time in shared memory.
# Outer-Product fasion: iterate through N's dim (128), after accessing each element - never use it again.
# 20.8[s]
@cuda.jit
def matmul_kernel_vanilla_shmem_outer_prod(A, C):
    tid = cuda.threadIdx.x
    if tid < CPT_PLUS_ONE:
        cpt = CPT + 1
        section_start = tid * cpt
    else:
        cpt = CPT
        section_start = CPT_PLUS_ONE * (cpt + 1) + (tid - CPT_PLUS_ONE) * cpt
    section_end = section_start + cpt - 1

    sA = cuda.shared.array(shape=(M, PREFETCH_WINDOW), dtype=numba.float64)

    for k in range(PREFETCH_WINDOW):
        if tid < M:
            sA[tid, k] = A[tid, k]
        else:
            break
    
    cuda.syncthreads()
    
    for k in range(N):
        for i in range(section_start, section_end+1):
            x = int((math.sqrt(8 * i + 1) - 1) / 2)
            y = int(i - x * (x + 1) / 2)    
            if k < PREFETCH_WINDOW:
                C[x, y] += sA[x, k] * sA[y, k]
            else:
                C[x, y] += A[x, k] * A[y, k]

    for i in range(section_start, section_end+1):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        C[y, x] = C[x, y]


# Improved load balancing + shared memory only on cols 0-6 of A
# Shared Memory in 2080TI is 48[KB]. Assuming 8[B] floats64, we can store floor(48*1024/(784*8)) cols of A at the same time in shared memory.
# Inner-Product fasion: iterate through X,Y's, for consecutive tasks, running on the same A rows - supposed to be worse than outer-product
# 11.3[s]
@cuda.jit
def matmul_kernel_vanilla_shmem_inner_prod(A, C):
    tid = cuda.threadIdx.x
    if tid < CPT_PLUS_ONE:
        cpt = CPT + 1
        section_start = tid * cpt
    else:
        cpt = CPT
        section_start = CPT_PLUS_ONE * (cpt + 1) + (tid - CPT_PLUS_ONE) * cpt
    section_end = section_start + cpt - 1

    sA = cuda.shared.array(shape=(M, PREFETCH_WINDOW), dtype=numba.float64)

    for k in range(PREFETCH_WINDOW):
        if tid < M:
            sA[tid, k] = A[tid, k]
        else:
            break
    
    cuda.syncthreads()
    
    for i in range(section_start, section_end+1):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)    
        for k in range(N):    
            if k < PREFETCH_WINDOW:
                C[x, y] += sA[x, k] * sA[y, k]
            else:
                C[x, y] += A[x, k] * A[y, k]
        C[y, x] = C[x, y]


@cuda.jit
def matmul_kernel(A, C):
    
    tid = cuda.threadIdx.x
    
    if tid < CPT_PLUS_ONE:
        cpt = CPT + 1
        section_start = tid * cpt
    else:
        cpt = CPT
        section_start = CPT_PLUS_ONE * (cpt + 1) + (tid - CPT_PLUS_ONE) * cpt
    section_end = section_start + cpt - 1
    
    sA = cuda.shared.array(shape=(M, PREFETCH_WINDOW), dtype=numba.float32)

    for k in range(0, N, PREFETCH_WINDOW):
        
        # Prefetching lock step - each thread is responsible to one row, brings <PREFETCH_WINDOW> items to shared memory
        for kk in range(k, min(k + PREFETCH_WINDOW, N)):
            if tid < M:  # Only <M> first threads participate in shared memory prefetching
                sA[tid, kk % PREFETCH_WINDOW] = A[tid, kk]

        cuda.syncthreads()
        
        # Compute lock step - each thread is responsible to a group of consecutive elements in the lower triangl result matrix
        # Only <PREFETCH_WINDOWS> partial sum portion of each vector inner product in being calculated
        for kk in range(k, min(k + PREFETCH_WINDOW, N)):
            for i in range(section_start, section_end+1):
                x = int((math.sqrt(8 * i + 1) - 1) / 2)
                y = int(i - x * (x + 1) / 2)
                C[x, y] += sA[x, kk % PREFETCH_WINDOW] * sA[y, kk % PREFETCH_WINDOW]

        cuda.syncthreads()

    # Fill the upper triangle matrix
    for i in range(section_start, section_end+1):
        x = int((math.sqrt(8 * i + 1) - 1) / 2)
        y = int(i - x * (x + 1) / 2)
        C[y, x] = C[x, y]


#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()
    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X,Xt)).repeat(3, 100))

    #print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    # print('Numpy:', timer(np.matmul, 2))
    # print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()

    X = np.reshape(np.arange(M*N), (M, N))
    Xt = X.copy().transpose()
    
    XXt_baseline = np.matmul(X, Xt)
    XXt = matmul_transpose_numba(X)
    print(not np.any(XXt_baseline - XXt))

    XXt = matmul_transpose_gpu(X)
    print(not np.any(XXt_baseline - XXt))
    print(np.nonzero(XXt_baseline - XXt))

    # import sys
    # np.set_printoptions(threshold=sys.maxsize)