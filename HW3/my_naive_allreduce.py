from time import time
import numpy as np
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

def allreduce(send, recv, comm, op):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    other_ranks = [i for i in range(size) if i != rank]
    
    reqs = []
    for other_rank in other_ranks:
        reqs.append(comm.Isend(send, other_rank))
    
    # for req in reqs:
    #     req.Wait()
    
    recvs_from_others = [np.empty_like(recv) for _ in range(size-1)]  # allocate storage for intermediate received arrays
    reqs = []
    for i, other_rank in enumerate(other_ranks):
        reqs.append(comm.Irecv(recvs_from_others[i], other_rank))
    
    for req in reqs:
        req.Wait()
    
    recv = send
    for recv_from_other in recvs_from_others:
        recv = op(recv, recv_from_other)
    
    # print('allreduce: {}/{}) send = \n----------\n{} recv = \n----------\n{}\n=========='.format(rank, size-1, send, recv))

    return recv


def sum(x,y):
    return x+y


def call_allreduce():
    MPI.Init()

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    send_data = np.reshape(np.arange(start=rank, stop=rank+1000000, dtype=np.float32), newshape=(1000,1000))
    recv_data = np.empty_like(send_data)

    recv_data = allreduce(send_data, recv_data, comm, sum)

    # print('{}/{}) send_data = \n----------\n{} recv_data = \n----------\n{}\n=========='.format(rank, size-1, send_data, recv_data))

    MPI.Finalize()

if __name__ == '__main__':    
    start = time()
    call_allreduce()
    stop = time()
    print(f'took {stop-start}[s]')