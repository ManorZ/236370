from time import time
import numpy as np
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


def ringallreduce(send, recv, comm, op):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

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

    send_slices = np.array_split(send, size)
    recv_slices = [np.empty_like(s) for s in send_slices]
    
    for i in range(2*size):
        # print('rank {} send slice {}: \n{}\n to rank {}'.format(rank, (rank-i)%size, send_slices[(rank-i)%size], (rank+1)%size))
        send_req = comm.Isend(send_slices[(rank-i)%size],   (rank+1)%size)
        # send_req.Wait()
        # recv_req = comm.Irecv(recv_slices[(rank-i-1)%size], (rank-1)%size)
        comm.Recv(recv_slices[(rank-i-1)%size], (rank-1)%size)
        # recv_req.Wait()
        # print('rank {} recv slice {}: \n{}\n from rank {}'.format(rank, (rank-i-1)%size, recv_slices[(rank-i-1)%size], (rank-1)%size))
        if i < size-1:  # Reduce Phase
            send_slices[(rank-i-1)%size] = op(recv_slices[(rank-i-1)%size], send_slices[(rank-i-1)%size])
        else:  # Shift Phase
            send_slices[(rank-i-1)%size] = recv_slices[(rank-i-1)%size]
        # print('rank {} updt slice {}: \n{}\n'.format(rank, (rank-i-1)%size, send_slices[(rank-i-1)%size]))
        send_req.Wait()

    recv = np.concatenate(send_slices, 0)

    # print('ringallreduce {}/{}) send = \n----------\n{} recv = \n----------\n{}\n=========='.format(rank, size-1, send, recv))

    return recv


def sum(x,y):
    return x+y


def call_ringallreduce():
    MPI.Init()

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    send_data = np.reshape(np.arange(start=rank, stop=rank+100000, dtype=np.float32), newshape=(500,200))
    recv_data = np.empty_like(send_data)

    recv_data = ringallreduce(send_data, recv_data, comm, sum)

    print('{}/{}) send_data = \n----------\n{} recv_data = \n----------\n{}\n=========='.format(rank, size-1, send_data, recv_data))

    MPI.Finalize()


if __name__ == '__main__':    
    start = time()
    call_ringallreduce()
    stop = time()
    print(f'took {stop-start}[s]')


        
