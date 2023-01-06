import math
import os
import pickle
import timeit
import numba
from numba import cuda
from numba import njit
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


TPB = (32,32)  # Threads per Block
M = None  # Image M dim
N = None  # Image N dim
m = None  # Kernel M dim
n = None  # Kernel N dim
mpad = None  # M dim padding
npad = None  # N dim padding
BPG = None  # Blocks per Grid
SAM = None  # Shmem M dim
SAN = None  # Shmem N dim

def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    global M
    global N
    global m
    global n
    global mpad
    global npad
    global BPG
    global SAM
    global SAN

    M = image.shape[0]
    N = image.shape[1]
    m = kernel.shape[0]
    n = kernel.shape[1]

    mpad = int(m/2)
    npad = int(n/2)
    
    BPG = (math.ceil(M/TPB[0]),math.ceil(N/TPB[1]))
    
    image = np.pad(image, ((mpad, mpad),(npad, npad)))
    
    SAM = TPB[0] + mpad*2
    SAN = TPB[1] + npad*2
    
    kernel_d = cuda.to_device(kernel)
    image_d = cuda.to_device(image)
    output_d = cuda.device_array((M,N), dtype=image.dtype)

    # correlation_cuda_kernel[BPG, TPB](kernel_d, image_d, output_d)
    # correlation_cuda_kernel_filter_shmem[BPG, TPB](kernel_d, image_d, output_d)
    # correlation_cuda_kernel_image_shmem[BPG, TPB](kernel_d, image_d, output_d)
    correlation_cuda_kernel_filter_and_image_shmem[BPG, TPB](kernel_d, image_d, output_d)

    output_h = output_d.copy_to_host()
    
    return output_h


@cuda.jit
def correlation_cuda_kernel(kernel, image, output):
    x,y = cuda.grid(2)

    if x >= output.shape[0] or y >= output.shape[1]:
        return

    for xx in range(-math.floor(m/2), math.ceil(m/2)):
        for yy in range(-math.floor(n/2), math.ceil(n/2)):
            output[x,y] += kernel[int(xx+math.floor(m/2)), int(yy+math.floor(n/2))] * image[int(x+xx)+mpad, int(y+yy)+npad]


@cuda.jit
def correlation_cuda_kernel_filter_shmem(kernel, image, output):
    x,y = cuda.grid(2)

    if x >= output.shape[0] or y >= output.shape[1]:
        return

    tidx = cuda.threadIdx.x
    tidy = cuda.threadIdx.y
    
    sK = cuda.shared.array(shape=(m, n), dtype=numba.float64)

    if tidx < m and tidy < n:  # first <m>X<n> threads copies the kernel to shmem as well
        sK[tidx,tidy] = kernel[tidx, tidy]
    
    cuda.syncthreads()

    for xx in range(-math.floor(m/2), math.ceil(m/2)):
        for yy in range(-math.floor(n/2), math.ceil(n/2)):
            output[x,y] += sK[int(xx+math.floor(m/2)), int(yy+math.floor(n/2))] * image[int(x+xx)+mpad, int(y+yy)+npad]


@cuda.jit
def correlation_cuda_kernel_image_shmem(kernel, image, output):
    x,y = cuda.grid(2)

    tidx = cuda.threadIdx.x
    tidy = cuda.threadIdx.y
    
    sI = cuda.shared.array(shape=(TPB[0], TPB[1]), dtype=numba.float64)
    
    sI[tidx,tidy] = image[x+mpad,y+npad]  # Each thread copies one pixel to shmem
    
    cuda.syncthreads()

    if x >= output.shape[0] or y >= output.shape[1]:
        return

    for xx in range(-math.floor(m/2), math.ceil(m/2)):
        for yy in range(-math.floor(n/2), math.ceil(n/2)):
            if tidx+xx > 0 and tidx+xx < sI.shape[0] and tidy+yy > 0 and tidy+yy < sI.shape[1]:
                output[x,y] += kernel[int(xx+math.floor(m/2)), int(yy+math.floor(n/2))] * sI[int(tidx+xx),int(tidy+yy)]
            else:
                output[x,y] += kernel[int(xx+math.floor(m/2)), int(yy+math.floor(n/2))] * image[int(x+xx)+mpad, int(y+yy)+npad]


@cuda.jit
def correlation_cuda_kernel_filter_and_image_shmem(kernel, image, output):
    x,y = cuda.grid(2)

    tidx = cuda.threadIdx.x
    tidy = cuda.threadIdx.y

    sK = cuda.shared.array(shape=(m, n), dtype=numba.float64)
    
    if tidx < m and tidy < n:  # first <m>X<n> threads copies the kernel to shmem as well
        sK[tidx,tidy] = kernel[tidx, tidy]

    cuda.syncthreads()
    
    sI = cuda.shared.array(shape=(TPB[0], TPB[1]), dtype=numba.float64)

    sI[tidx,tidy] = image[x+mpad,y+npad]  # Each thread copies one pixel to shmem
    
    cuda.syncthreads()

    if x >= output.shape[0] or y >= output.shape[1]:
        return

    for xx in range(-math.floor(m/2), math.ceil(m/2)):
        for yy in range(-math.floor(n/2), math.ceil(n/2)):
            if tidx+xx > 0 and tidx+xx < sI.shape[0] and tidy+yy > 0 and tidy+yy < sI.shape[1]:
                output[x,y] += sK[int(xx+math.floor(m/2)), int(yy+math.floor(n/2))] * sI[int(tidx+xx),int(tidy+yy)]
            else:
                output[x,y] += sK[int(xx+math.floor(m/2)), int(yy+math.floor(n/2))] * image[int(x+xx)+mpad, int(y+yy)+npad]


@njit
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''

    m = kernel.shape[0]
    n = kernel.shape[1]
    M = image.shape[0]
    N = image.shape[1]

    mpad = int(m/2)
    npad = int(n/2)
    
    output = np.zeros_like(image)
    # image = np.pad(image, ((mpad, mpad),(npad, npad)))
    image_pad = np.zeros((M + mpad*2, N + npad*2))
    image_pad[mpad:-mpad, npad:-npad] = image
    for y in range(mpad, mpad+M):
        for x in range(npad, npad+N):
            output[y-mpad, x-npad] = (kernel * image_pad[y-mpad:y+mpad+1,x-npad:x+npad+1]).sum()
    
    return output
    

def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    # your calculations

    raise NotImplementedError("To be implemented")


def load_image(): 
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image, fname):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.savefig(fname)
    plt.show()


def correlation_cpu(kernel, image):
    return convolve2d(image, np.flipud(np.fliplr(kernel)), mode='same')


def timer(kernel, f):
    return min(timeit.Timer(lambda: f(kernel, image)).repeat(20, 1))


if __name__ == '__main__':
    # 7X7
    edge_kernel = np.array(
        [
            [-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18],
            [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
            [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
            [-3/9, -2/4, -1/1, 0, 1/1, 2/4, 3/9],
            [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
            [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
            [-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18]
        ]
    )
    flipped_edge_kernel = np.array(
        [
            [3/18, 2/13, 1/10, 0, -1/10, -2/13, -3/18],
            [3/13, 2/8, 1/5, 0, -1/5, -2/8, -3/13],
            [3/10, 2/5, 1/2, 0, -1/2, -2/5, -3/10],
            [3/9, 2/4, 1/1, 0, -1/1, -2/4, -3/9],
            [3/10, 2/5, 1/2, 0, -1/2, -2/5, -3/10],
            [3/13, 2/8, 1/5, 0, -1/5, -2/8, -3/13],
            [3/18, 2/13, 1/10, 0, -1/10, -2/13, -3/18]
        ]
    )
    # 5X5
    blur_kernel = np.array(
        [
            [1/25, 1/25, 1/25, 1/25, 1/25],
            [1/25, 1/25, 1/25, 1/25, 1/25],
            [1/25, 1/25, 1/52, 1/25, 1/25],
            [1/25, 1/25, 1/25, 1/25, 1/25],
            [1/25, 1/25, 1/25, 1/25, 1/25]
        ]
    )
    # 3X3
    shapen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]
    )
    # 3X5
    non_symmetric_kernel = np.array(
        [
            [0, -1, -2, 1, 0],
            [1,  0, -1, 2, 1],
            [0, -1, -2, 1, 0],
        ]
    )
    
    image = load_image()

    # image_scipy = convolve2d(image, np.flipud(np.fliplr(non_symmetric_kernel)), mode='same')
    # image_numba = correlation_numba(non_symmetric_kernel, image)
    # image_cuda = correlation_gpu(non_symmetric_kernel, image)
    # print('non_symmetric_kernel numba vs. scipy:', np.linalg.norm(image_numba-image_scipy))
    # print('non_symmetric_kernel cuda vs. scipy:', np.linalg.norm(image_cuda-image_scipy))

    # scipy_time = timer(non_symmetric_kernel, correlation_cpu)
    # print('non_symmetric_kernel scipy time:', scipy_time)
    # numba_time = timer(non_symmetric_kernel, correlation_numba)
    # print('non_symmetric_kernel numba time:', numba_time)
    # gpu_time = timer(non_symmetric_kernel, correlation_gpu)
    # print('non_symmetric_kernel cuda time:', gpu_time)

    # gpu_scipy_speedup = scipy_time / gpu_time
    # numba_scipy_speedup = scipy_time / numba_time
    # print('non_symmetric_kernel numba-scipy speedup:', numba_scipy_speedup)
    # print('non_symmetric_kernel cuda-scipy speedup:', gpu_scipy_speedup)


    # image_scipy = convolve2d(image, np.flipud(np.fliplr(shapen_kernel)), mode='same')
    # image_numba = correlation_numba(shapen_kernel, image)
    # image_cuda = correlation_gpu(shapen_kernel, image)
    # print('shapen_kernel numba vs. scipy:', np.linalg.norm(image_numba-image_scipy))
    # print('shapen_kernel cuda vs. scipy:', np.linalg.norm(image_cuda-image_scipy))

    # scipy_time = timer(shapen_kernel, correlation_cpu)
    # print('shapen_kernel scipy time:', scipy_time)
    # numba_time = timer(shapen_kernel, correlation_numba)
    # print('shapen_kernel numba time:', numba_time)
    # gpu_time = timer(shapen_kernel, correlation_gpu)
    # print('shapen_kernel cuda time:', gpu_time)

    # gpu_scipy_speedup = scipy_time / gpu_time
    # numba_scipy_speedup = scipy_time / numba_time
    # print('shapen_kernel numba-scipy speedup:', numba_scipy_speedup)
    # print('shapen_kernel cuda-scipy speedup:', gpu_scipy_speedup)

    
    # image_scipy = convolve2d(image, np.flipud(np.fliplr(blur_kernel)), mode='same')
    # image_numba = correlation_numba(blur_kernel, image)
    # image_cuda = correlation_gpu(blur_kernel, image)
    # print('blur_kernel numba vs. scipy:', np.linalg.norm(image_numba-image_scipy))
    # print('blur_kernel cuda vs. scipy:', np.linalg.norm(image_cuda-image_scipy))

    # scipy_time = timer(blur_kernel, correlation_cpu)
    # print('blur_kernel scipy time:', scipy_time)
    # numba_time = timer(blur_kernel, correlation_numba)
    # print('blur_kernel numba time:', numba_time)
    # gpu_time = timer(blur_kernel, correlation_gpu)
    # print('blur_kernel cuda time:', gpu_time)

    # gpu_scipy_speedup = scipy_time / gpu_time
    # numba_scipy_speedup = scipy_time / numba_time
    # print('blur_kernel numba-scipy speedup:', numba_scipy_speedup)
    # print('blur_kernel cuda-scipy speedup:', gpu_scipy_speedup)


    # image_scipy = convolve2d(image, np.flipud(np.fliplr(edge_kernel)), mode='same')
    # image_numba = correlation_numba(edge_kernel, image)
    # image_cuda = correlation_gpu(edge_kernel, image)
    # print('edge_kernel numba vs. scipy:', np.linalg.norm(image_numba-image_scipy))
    # print('edge_kernel cuda vs. scipy:', np.linalg.norm(image_cuda-image_scipy))

    scipy_time = timer(edge_kernel, correlation_cpu)
    print('edge_kernel scipy time:', scipy_time)
    numba_time = timer(edge_kernel, correlation_numba)
    print('edge_kernel numba time:', numba_time)
    gpu_time = timer(edge_kernel, correlation_gpu)
    print('edge_kernel cuda time:', gpu_time)

    gpu_scipy_speedup = scipy_time / gpu_time
    numba_scipy_speedup = scipy_time / numba_time
    print('edge_kernel numba-scipy speedup:', numba_scipy_speedup)
    print('edge_kernel cuda-scipy speedup:', gpu_scipy_speedup)