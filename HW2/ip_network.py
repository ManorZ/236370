import sys
import math
import time
import matplotlib.pyplot as plt

from network import *
from preprocessor import *
from my_queue import MyQueue

def plot_minibatch(data, labels, indices, batch_size, k):
    
    def one_hot_to_number(x):
        return np.where(x == 1)[0].item()
    
    rows = math.ceil(math.sqrt(batch_size))
    cols = math.ceil(math.sqrt(batch_size))
    fig, axes = plt.subplots(rows, cols)
    plt.tight_layout()
    for i,ax in enumerate(axes.flat):
        ax.imshow(data[indices[i]].reshape((28,28)))
        ax.set_title(one_hot_to_number(labels[indices[i]]))
    plt.show()
    fig.savefig(f'original_minibatch{k}')

class IPNeuralNetwork(NeuralNetwork):
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        self.jobs = JoinableQueue()
        self.result = Queue()
        # self.result = MyQueue()

        self.num_workers = multiprocessing.cpu_count() * 2
        workers = [ Worker(self.jobs, self.result, training_data, self.mini_batch_size) for i in range(self.num_workers) ]

        for w in workers:
            w.start()
		
        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)
        
        for _ in range(self.num_workers):
            self.jobs.put(None)
        
        self.jobs.join()
        
        [ w.join() for w in workers ]


    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        for k in range(self.number_of_batches):
            indices = random.sample(range(0, data.shape[0]), batch_size)
            
            job = {'batch ID': k, 'indices ID': indices}
            self.jobs.put(job)
            
        batches = []
        for k in range(self.number_of_batches):
            batches.append(self.result.get())
            
        return batches
    
