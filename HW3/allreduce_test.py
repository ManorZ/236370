import sys
from time import time
import numpy as np
from my_naive_allreduce import *
from my_ring_allreduce import *
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

MPI.Init()

la_comm = MPI.COMM_WORLD
ma_rank = la_comm.Get_rank()
la_size = la_comm.Get_size()

def _op(x, y):
    return x+y


for size in [2**12, 2**13, 2**14, 2**15, 2**16]:
    print("array size:", size)
    data = np.random.rand(size)
    res1 = np.zeros_like(data)
    res2 = np.zeros_like(data)
    start1 = time()
    allreduce(data, res1, la_comm, _op)
    end1 = time()
    naive_time = end1-start1
    print(f"naive impl time:   {naive_time:.6f}")
    start1 = time()
    ringallreduce(data, res2, la_comm, _op)
    end1 = time()
    ring_time=end1-start1
    print(f"ring impl time:    {ring_time:.6f}")
    ring_speedup = naive_time / ring_time
    print(f"ring impl speedup: {ring_speedup:.2f}")
    print('-'*50)
    assert np.allclose(res1, res2)

MPI.Finalize()
