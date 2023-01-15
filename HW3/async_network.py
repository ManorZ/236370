from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time

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

        print(f'size={self.size}, #workers={self.num_workers} #masters={self.num_masters}')

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
        # TODO: add your code

        self.number_of_batches = self.number_of_batches // self.num_workers

        print(f'worker {self.rank} run {self.number_of_batches} minibatches')

        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for minibatch, (x, y) in enumerate(mini_batches):
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters 
                # TODO: add your code

                master_idx = 0
                for layer_idx in range(self.num_layers):
                    # print(f'{epoch},{minibatch:<4}) worker {self.rank} send nabla_w[{layer_idx}] to master {master_idx}: np.any(np.isnan(nabla_w[{layer_idx}])) = {np.any(np.isnan(nabla_w[layer_idx]))}')
                    self.comm.Isend(nabla_w[layer_idx], master_idx)
                    master_idx = (master_idx+1)%self.num_masters
                
                master_idx = 0
                for layer_idx in range(self.num_layers):
                    # print(f'{epoch},{minibatch:<4}) worker {self.rank} send nabla_b[{layer_idx}] to master {master_idx}: np.any(np.isnan(nabla_b[{layer_idx}])) = {np.any(np.isnan(nabla_b[layer_idx]))}')
                    self.comm.Isend(nabla_b[layer_idx], master_idx)
                    master_idx = (master_idx+1)%self.num_masters

                # recieve new self.weight and self.biases values from masters
                # TODO: add your code

                master_idx = 0
                # recv_w_reqs = []
                for layer_idx in range(self.num_layers):
                    # recv_w_reqs.append(self.comm.Irecv(self.weights[layer_idx], master_idx))
                    recv_w_req = self.comm.Irecv(self.weights[layer_idx], master_idx)
                    recv_w_req.Wait()
                    master_idx = (master_idx+1)%self.num_masters
                
                # for recv_w_req in recv_w_reqs:
                #     recv_w_req.Wait()
                
                master_idx = 0
                # recv_b_reqs = []
                for layer_idx in range(self.num_layers):
                    # recv_b_reqs.append(self.comm.Irecv(self.biases[layer_idx], master_idx))
                    recv_b_req = self.comm.Irecv(self.biases[layer_idx], master_idx)
                    recv_b_req.Wait()
                    master_idx = (master_idx+1)%self.num_masters
                
                # for recv_w_req in recv_w_reqs:
                #     recv_w_req.Wait()

                # for recv_b_req in recv_b_reqs:
                #     recv_b_req.Wait()
                
                # NOTE: debug print
                # master_idx = 0
                # for layer_idx in range(self.num_layers):
                    # print(f'{epoch},{minibatch:<4}) worker {self.rank} weights[{layer_idx}] after recv from master {master_idx}: np.any(np.isnan(weights[{layer_idx}])) = {np.any(np.isnan(self.weights[layer_idx]))}')
                    # print(f'{epoch},{minibatch:<4}) worker {self.rank} biases[{layer_idx}]  after recv from master {master_idx}: np.any(np.isnan(biases[{layer_idx}]))  = {np.any(np.isnan(self.biases[layer_idx]))}')
                    # master_idx = (master_idx+1)%self.num_masters
                
                

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
        
        self.number_of_batches = (self.number_of_batches // self.num_workers) * self.num_workers

        print(f'master {self.rank} run {self.number_of_batches} minibatches')

        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):

                # master iterates through the effective number of batches - each worker gets a subset of it.
                # since the first M ranks are masters, the next W ranks are workers
                worker_idx = batch % self.num_workers + self.num_masters

                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                # TODO: add your code

                # recv_w_reqs = []
                for i in range(len(nabla_w)):
                    # recv_w_reqs.append(self.comm.Irecv(nabla_w[i], worker_idx))
                    recv_w_req = self.comm.Irecv(nabla_w[i], worker_idx)
                    recv_w_req.Wait()
                    
                # for recv_w_req in recv_w_reqs:
                #     recv_w_req.Wait()
                
                # recv_b_reqs = []
                for i in range(len(nabla_b)):
                    # recv_b_reqs.append(self.comm.Irecv(nabla_b[i], worker_idx))
                    recv_b_req = self.comm.Irecv(nabla_b[i], worker_idx)
                    recv_b_req.Wait()
                
                # for recv_w_req in recv_w_reqs:
                #     recv_w_req.Wait()

                # for recv_b_req in recv_b_reqs:
                #     recv_b_req.Wait()
                
                # NOTE: debug print
                # for i in range(len(nabla_w)):
                #     print(f'{epoch},{batch:<4}) master {self.rank} nabla_w[{i}] after recv from worker {worker_idx}: np.any(np.isnan(nabla_w[{i}])) = {np.any(np.isnan(nabla_w[i]))}')
                
                # for i in range(len(nabla_b)):
                #     print(f'{epoch},{batch:<4}) master {self.rank} nabla_b[{i}] after recv from worker {worker_idx}: np.any(np.isnan(nabla_b[{i}])) = {np.any(np.isnan(nabla_b[i]))}')
                
                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db
                
                # send new values (of layers in charge)
                # TODO: add your code
                
                for i, dw in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w):
                    # print(f'{epoch},{batch:<4}) master {self.rank} send weights[{i}] to worker {worker_idx}: np.any(np.isnan(weights[{i}])) = {np.any(np.isnan(self.weights[i]))}')
                    self.comm.Isend(self.weights[i], worker_idx)
                
                for i, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_b):
                    # print(f'{epoch},{batch:<4}) master {self.rank} send biases[{i}]  to worker {worker_idx}: np.any(np.isnan(biases[{i}]))  = {np.any(np.isnan(self.biases[i]))}')
                    self.comm.Isend(self.biases[i], worker_idx)
                
            # self.print_progress(validation_data, epoch)  # NOTE: no point to evaluate progress during training when each master has only part of the layers - the accuracy will always be minimal.

            # gather relevant weight and biases to process 0
            # TODO: add your code
            # NOTE: this was outside the for epoch loop but I think it should be here! This is how it works for real systems. Every epoch, the main parameters server is synced with all others.
            # if self.rank != 0:
            #     # send all layers under this master responsibility
            #     for i in range(self.rank, self.num_layers, self.num_masters):
            #         self.comm.Isend(self.weights[i], 0)
            #     for i in range(self.rank, self.num_layers, self.num_masters):
            #         self.comm.Isend(self.biases[i], 0)
            # if self.rank == 0:
            #     # iterate through all other masters - receive from each one his layers
            #     for master_idx in range(1, self.num_masters):
            #         for i in range(master_idx, self.num_layers, self.num_masters):
            #             recv_w_req = self.comm.Irecv(self.weights[i], master_idx)
            #             recv_w_req.Wait()
            #         for i in range(master_idx, self.num_layers, self.num_masters):
            #             recv_b_req = self.comm.Irecv(self.biases[i], master_idx)
            #             recv_b_req.Wait()

        # gather relevant weight and biases to process 0
        # TODO: add your code
        if self.rank != 0:
            # send all layers under this master responsibility
            for i in range(self.rank, self.num_layers, self.num_masters):
                self.comm.Isend(self.weights[i], 0)
            for i in range(self.rank, self.num_layers, self.num_masters):
                self.comm.Isend(self.biases[i], 0)
        if self.rank == 0:
            # iterate through all other masters - receive from each one his layers
            for master_idx in range(1, self.num_masters):
                for i in range(master_idx, self.num_layers, self.num_masters):
                    recv_w_req = self.comm.Irecv(self.weights[i], master_idx)
                    recv_w_req.Wait()
                for i in range(master_idx, self.num_layers, self.num_masters):
                    recv_b_req = self.comm.Irecv(self.biases[i], master_idx)
                    recv_b_req.Wait()
        