from network import *
from my_ring_allreduce import *
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class SynchronicNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):

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
                # nabla_w = [np.zeros_like(w) for w in self.weights]
                # nabla_b = [np.zeros_like(b) for b in self.biases]
                
                for ma_nab_w, ma_nab_b, nab_w, nab_b in zip(ma_nabla_w, ma_nabla_b, nabla_w, nabla_b):
                    comm.Allreduce(ma_nab_w, nab_w, op=MPI.SUM)
                    comm.Allreduce(ma_nab_b, nab_b, op=MPI.SUM)
                
                # TODO: I would be glad to find a way to pack list of numpy array and send in one piece...
                # comm.Allreduce(ma_nabla_w, nabla_w, op=MPI.SUM)
                # comm.Allreduce(ma_nabla_b, nabla_b, op=MPI.SUM)

                # NOTE: although I should have averaged the grads (tutorial 7, slide 13), I noticed that in this simple case, it works also with sum only.
                # for w, b in zip(nabla_w, nabla_b):
                #     w = w/size
                #     b = b/size

                # calculate work
                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]

            self.print_progress(validation_data, epoch)

        MPI.Finalize()
