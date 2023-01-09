from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time, sleep

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        # setting number of workers and masters
        self.num_masters = number_of_masters

    def fit(self, training_data, validation_data=None):
        # MPI setup
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = self.size - self.num_masters

        self.layers_per_master = self.num_layers // self.num_masters

        # split up work
        if self.rank < self.num_masters:
            self.do_master(validation_data)
        else:
            self.do_worker(training_data)

        # when all is done
        self.comm.Barrier()
        MPI.Finalize()

    def do_worker(self, training_data):
        """
        worker functionality
        :param training_data: a tuple of data and labels to train the NN with
        """
        # setting up the number of batches the worker should do every epoch
        self.number_of_batches = self.number_of_batches // self.num_workers

        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for minibatch_idx, (x, y) in enumerate(mini_batches):
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters 
                # TODO: add your code
            
                # print(f'worker {self.rank} send nabla_w[0] to master 0: {nabla_w[0][0]:.4f} ({nabla_w[0].shape})')
                self.comm.Isend(nabla_w[0], 0)
                # print(f'worker {self.rank} send nabla_w[1] to master 0: {nabla_w[1][0,0]:.4f}...{nabla_w[1][-1,-1]:.4f} ({nabla_w[1].shape})')
                self.comm.Isend(nabla_w[1], 0)
                # print(f'worker {self.rank} send nabla_w[2] to master 0: {nabla_w[2][0,0]:.4f}...{nabla_w[2][-1,-1]:.4f} ({nabla_w[2].shape})')
                self.comm.Isend(nabla_w[2], 0)
                # print(f'worker {self.rank} send nabla_w[3] to Master 0: {nabla_w[3][0,0]:.4f}...{nabla_w[3][-1,-1]:.4f} ({nabla_w[3].shape})')
                self.comm.Isend(nabla_w[3], 0)

                # print(f'worker {self.rank} send nabla_b[0] to master 0: {nabla_b[0][0]:.4f} ({nabla_b[0].shape})')
                self.comm.Isend(nabla_b[0], 0)
                # print(f'worker {self.rank} send nabla_b[1] to Master 0: {nabla_b[1][0]:.4f}...{nabla_b[1][-1]:.4f} ({nabla_b[1].shape})')
                self.comm.Isend(nabla_b[1], 0)
                # print(f'worker {self.rank} send nabla_b[2] to master 0: {nabla_b[2][0]:.4f}...{nabla_b[2][-1]:.4f} ({nabla_b[2].shape})')
                self.comm.Isend(nabla_b[2], 0)
                # print(f'worker {self.rank} send nabla_b[3] to Master 0: {nabla_b[3][0]:.4f}...{nabla_b[3][-1]:.4f} ({nabla_b[3].shape})')
                self.comm.Isend(nabla_b[3], 0)

                # recieve new self.weight and self.biases values from masters
                # TODO: add your code
                
                # print(f'worker {self.rank} weights[0] before recv from master 0: {self.weights[0][0]:.4f} ({self.weights[0].shape})')
                # print(f'worker {self.rank} weights[1] before recv from Master 0: {self.weights[1][0,0]:.4f}...{self.weights[1][-1,-1]:.4f} ({self.weights[1].shape})')
                # print(f'worker {self.rank} biases[0]  before recv from master 0: {self.biases[0][0]:.4f} ({self.biases[0].shape})')
                # print(f'worker {self.rank} biases[1]  before recv from Master 0: {self.biases[1][0]:.4f}...{self.biases[1][-1]:.4f} ({self.biases[1].shape})')
                # print(f'worker {self.rank} weights[2] before recv from master 0: {self.weights[2][0,0]:.4f}...{self.weights[2][-1,-1]:.4f} ({self.weights[2].shape})')
                # print(f'worker {self.rank} weights[3] before recv from Master 0: {self.weights[3][0,0]:.4f}...{self.weights[3][-1,-1]:.4f} ({self.weights[3].shape})')
                # print(f'worker {self.rank} biases[2]  before recv from master 0: {self.biases[2][0]:.4f}...{self.biases[2][-1]:.4f} ({self.biases[2].shape})')
                # print(f'worker {self.rank} biases[3]  before recv from Master 0: {self.biases[3][0]:.4f}...{self.biases[3][-1]:.4f} ({self.biases[3].shape})')
                recv_w_req0 = self.comm.Irecv(self.weights[0], 0)
                recv_w_req1 = self.comm.Irecv(self.weights[1], 0)
                recv_w_req2 = self.comm.Irecv(self.weights[2], 0)
                recv_w_req3 = self.comm.Irecv(self.weights[3], 0)

                recv_b_req0 = self.comm.Irecv(self.biases[0], 0)
                recv_b_req1 = self.comm.Irecv(self.biases[1], 0)
                recv_b_req2 = self.comm.Irecv(self.biases[2], 0)
                recv_b_req3 = self.comm.Irecv(self.biases[3], 0)
                
                recv_w_req0.Wait()
                recv_w_req1.Wait()
                recv_w_req2.Wait()
                recv_w_req3.Wait()

                recv_b_req0.Wait()
                recv_b_req1.Wait()
                recv_b_req2.Wait()
                recv_b_req3.Wait()

                # print(f'worker {self.rank} weights[0] after  recv from master 0: {self.weights[0][0]:.4f} ({self.weights[0].shape})')
                # print(f'worker {self.rank} weights[1] after  recv from Master 0: {self.weights[1][0,0]:.4f}...{self.weights[1][-1,-1]:.4f} ({self.weights[1].shape})')
                # print(f'worker {self.rank} weights[2] after  recv from master 0: {self.weights[2][0,0]:.4f}...{self.weights[2][-1,-1]:.4f} ({self.weights[2].shape})')
                # print(f'worker {self.rank} weights[3] after  recv from Master 0: {self.weights[3][0,0]:.4f}...{self.weights[3][-1,-1]:.4f} ({self.weights[3].shape})')
                # print(f'worker {self.rank} biases[0]  after  recv from master 0: {self.biases[0][0]:.4f} ({self.biases[0].shape})')
                # print(f'worker {self.rank} biases[1]  after  recv from Master 0: {self.biases[1][0]:.4f}...{self.biases[1][-1]:.4f} ({self.biases[1].shape})')
                # print(f'worker {self.rank} biases[2]  after  recv from master 0: {self.biases[2][0]:.4f}...{self.biases[2][-1]:.4f} ({self.biases[2].shape})')
                # print(f'worker {self.rank} biases[3]  after  recv from Master 0: {self.biases[3][0]:.4f}...{self.biases[3][-1]:.4f} ({self.biases[3].shape})')
                
    def do_master(self, validation_data):
        """
        master functionality
        :param validation_data: a tuple of data and labels to train the NN with
        """
        # setting up the layers this master does
        nabla_w = []
        nabla_b = []
        for i in range(self.rank, self.num_layers, self.num_masters):
            nabla_w.append(np.zeros_like(self.weights[i]))
            nabla_b.append(np.zeros_like(self.biases[i]))
        
        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):

                # master iterates through the effective number of batches.
                worker_idx = batch % self.num_workers + self.num_masters
                
                # print(f'Master 0 nabla_w[0] before recv from Worker {worker_idx}: {nabla_w[0][0]:.4f} ({nabla_w[0].shape})')
                recv_nab_w_req0 = self.comm.Irecv(nabla_w[0], worker_idx)
                recv_nab_w_req0.Wait()
                # print(f'Master 0 nabla_w[0] after  recv from Worker {worker_idx}: {nabla_w[0][0]:.4f} ({nabla_w[0].shape})')
                # print(f'Master 0 nabla_w[1] before recv from Worker {worker_idx}: {nabla_w[1][0,0]:.4f}...{nabla_w[1][-1,-1]:.4f} ({nabla_w[1].shape})')  
                recv_nab_w_req1 = self.comm.Irecv(nabla_w[1], worker_idx)
                recv_nab_w_req1.Wait()
                # print(f'Master 0 nabla_w[1] after  recv from Worker {worker_idx}: {nabla_w[1][0,0]:.4f}...{nabla_w[1][-1,-1]:.4f} ({nabla_w[1].shape})')
                # print(f'Master 0 nabla_w[2] before recv from Worker {worker_idx}: {nabla_w[2][0,0]:.4f}...{nabla_w[2][-1,-1]:.4f} ({nabla_w[2].shape})')  
                recv_nab_w_req2 = self.comm.Irecv(nabla_w[2], worker_idx)
                recv_nab_w_req2.Wait()
                # print(f'Master 0 nabla_w[2] after  recv from Worker {worker_idx}: {nabla_w[2][0,0]:.4f}...{nabla_w[2][-1,-1]:.4f} ({nabla_w[2].shape})')
                # print(f'Master 0 nabla_w[3] before recv from Worker {worker_idx}: {nabla_w[3][0,0]:.4f}...{nabla_w[3][-1,-1]:.4f} ({nabla_w[3].shape})')  
                recv_nab_w_req3 = self.comm.Irecv(nabla_w[3], worker_idx)
                recv_nab_w_req3.Wait()
                # print(f'Master 0 nabla_w[3] after  recv from Worker {worker_idx}: {nabla_w[3][0,0]:.4f}...{nabla_w[3][-1,-1]:.4f} ({nabla_w[3].shape})')

                # print(f'Master 0 nabla_b[0] before recv from Worker {worker_idx}: {nabla_b[0][0]:.4f} ({nabla_b[0].shape})')
                recv_nab_b_req0 = self.comm.Irecv(nabla_b[0], worker_idx)
                recv_nab_b_req0.Wait()
                # print(f'Master 0 nabla_b[0] after  recv from Worker {worker_idx}: {nabla_b[0][0]:.4f} ({nabla_b[0].shape})')
                # print(f'Master 0 nabla_b[1] before recv from Worker {worker_idx}: {nabla_b[1][0]:.4f}...{nabla_b[1][-1]:.4f} ({nabla_b[1].shape})')  
                recv_nab_b_req1 = self.comm.Irecv(nabla_b[1], worker_idx)
                recv_nab_b_req1.Wait()
                # print(f'Master 0 nabla_b[1] after  recv from Worker {worker_idx}: {nabla_b[1][0]:.4f}...{nabla_b[1][-1]:.4f} ({nabla_b[1].shape})')
                # print(f'Master 0 nabla_b[2] before recv from Worker {worker_idx}: {nabla_b[2][0]:.4f}...{nabla_b[2][-1]:.4f} ({nabla_b[2].shape})')  
                recv_nab_b_req2 = self.comm.Irecv(nabla_b[2], worker_idx)
                recv_nab_b_req2.Wait()
                # print(f'Master 0 nabla_b[2] after  recv from Worker {worker_idx}: {nabla_b[2][0]:.4f}...{nabla_b[2][-1]:.4f} ({nabla_b[2].shape})')
                # print(f'Master 0 nabla_b[3] before recv from Worker {worker_idx}: {nabla_b[3][0]:.4f}...{nabla_b[3][-1]:.4f} ({nabla_b[3].shape})')  
                recv_nab_b_req3 = self.comm.Irecv(nabla_b[3], worker_idx)
                recv_nab_b_req3.Wait()
                # print(f'Master 0 nabla_b[3] after  recv from Worker {worker_idx}: {nabla_b[3][0]:.4f}...{nabla_b[3][-1]:.4f} ({nabla_b[3].shape})')
                
                self.weights[0] = self.weights[0] - self.eta * nabla_w[0]
                self.weights[1] = self.weights[1] - self.eta * nabla_w[1]
                self.weights[2] = self.weights[2] - self.eta * nabla_w[2]
                self.weights[3] = self.weights[3] - self.eta * nabla_w[3]
                
                self.biases[0] = self.biases[0] - self.eta * nabla_b[0]
                self.biases[1] = self.biases[1] - self.eta * nabla_b[1]
                self.biases[2] = self.biases[2] - self.eta * nabla_b[2]
                self.biases[3] = self.biases[3] - self.eta * nabla_b[3]

                # print(f'Master 0 send weights[0] to Worker {worker_idx}: {self.weights[0][0]:.4f} ({self.weights[0].shape})')
                self.comm.Isend(self.weights[0], worker_idx)
                # print(f'Master 0 send weights[1] to Worker {worker_idx}: {self.weights[1][0,0]:.4f}...{self.weights[1][-1,-1]:.4f} ({self.weights[1].shape})')
                self.comm.Isend(self.weights[1], worker_idx)
                # print(f'Master 0 send weights[2] to Worker {worker_idx}: {self.weights[2][0,0]:.4f}...{self.weights[2][-1,-1]:.4f} ({self.weights[2].shape})')
                self.comm.Isend(self.weights[2], worker_idx)
                # print(f'Master 0 send weights[3] to Worker {worker_idx}: {self.weights[3][0,0]:.4f}...{self.weights[3][-1,-1]:.4f} ({self.weights[3].shape})')
                self.comm.Isend(self.weights[3], worker_idx)

                # print(f'Master 0 send biases[0]  to Worker {worker_idx}: {self.biases[0][0]:.4f} ({self.biases[0].shape})')
                self.comm.Isend(self.biases[0], worker_idx)                    
                # print(f'Master 0 send biases[1]  to Worker {worker_idx}: {self.biases[1][0]:.4f}...{self.biases[1][-1]:.4f} ({self.biases[1].shape})')
                self.comm.Isend(self.biases[1], worker_idx)
                # print(f'Master 0 send biases[2]  to Worker {worker_idx}: {self.biases[2][0]:.4f}...{self.biases[2][-1]:.4f} ({self.biases[2].shape})')
                self.comm.Isend(self.biases[2], worker_idx)
                # print(f'Master 0 send biases[3]  to Worker {worker_idx}: {self.biases[3][0]:.4f}...{self.biases[3][-1]:.4f} ({self.biases[3].shape})')
                self.comm.Isend(self.biases[3], worker_idx)

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        # TODO: add your code
        


            
