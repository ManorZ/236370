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

        # replace the 1st dummy layer of weights with the same array by with dtype=np.float64,
        # due to some weird problem in comm.Allreduce: ValueError: mismatch in send and receive MPI datatypes
        self.weights[0] = [np.array([0], dtype=np.float64)]

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
    
    def back_prop(self, y):
        nabla_b = zeros_biases(self.sizes)
        # replace the 1st dummy layer of weights grads with the same array by with dtype=np.float64,
        # due to some weird problem in comm.Allreduce: ValueError: mismatch in send and receive MPI datatypes
        # nabla_w = [np.array([0])] + zeros_weights(self.sizes)
        nabla_w = [np.array([0], dtype=np.float64)] + zeros_weights(self.sizes)
        error = (self.activations[-1] - y) * sigmoid_prime(self.zs[-1])
        nabla_b[-1] = np.sum(error, axis=0)
        nabla_w[-1] = self.matmul(self.activations[-2].T, error)

        for l in range(self.num_layers - 2, 0, -1):
            error = self.matmul(error, self.weights[l + 1].T) * sigmoid_prime(self.zs[l])
            nabla_b[l] = np.sum(error, axis=0)
            nabla_w[l] = self.matmul(self.activations[l - 1].T, error)

        return nabla_b, nabla_w
