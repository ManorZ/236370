from network import *
from my_ring_allreduce import *
import my_naive_allreduce
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class SynchronicNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):

        def sum_op(x,y):
            return x+y
        
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for epoch in range(self.epochs):

            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size // size)

            nabla_w = [np.zeros_like(w) for w in self.weights]
            nabla_b = [np.zeros_like(b) for b in self.biases]

            for x, y in mini_batches:
                # doing props
                self.forward_prop(x)
                ma_nabla_b, ma_nabla_w = self.back_prop(y)

                # NOTE: moved outside the loop for better memory allocation efficiency
                # summing all ma_nabla_b and ma_nabla_w to nabla_w and nabla_b
                # nabla_w = []
                # nabla_b = []
                
                for i, (ma_nab_w, ma_nab_b, nab_w, nab_b) in enumerate(zip(ma_nabla_w, ma_nabla_b, nabla_w, nabla_b)):
                    
                    # NOTE: changing in-place does not work for some reason... I'm too tired to figure our why... :0
                    nabla_w[i] = ringallreduce(ma_nab_w, nab_w, comm, sum_op)
                    nabla_b[i] = ringallreduce(ma_nab_b, nab_b, comm, sum_op)
                    
                    # NOTE: for comparisons: Ring AllReduce acheives 1.5x better perf than Naive AllReduce
                    # nabla_w[i] = my_naive_allreduce.allreduce(ma_nab_w, nab_w, comm, sum_op)
                    # nabla_b[i] = my_naive_allreduce.allreduce(ma_nab_b, nab_b, comm, sum_op)

                # calculate work
                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]

            self.print_progress(validation_data, epoch)

        MPI.Finalize()
