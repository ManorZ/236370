import multiprocessing
from multiprocessing import JoinableQueue, Queue, Process
from typing import Tuple
import numpy as np
from scipy import ndimage


class Worker(Process):
    
    def __init__(self, jobs: JoinableQueue, result: Queue, training_data: Tuple[np.ndarray, np.ndarray], batch_size: int):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
        self.jobs = jobs
        self.result = result
        self.training_data = training_data
        self.batch_size = batch_size

        self.buffer = (np.empty((batch_size,784), dtype=np.float32), np.empty((batch_size, 10), dtype=np.float64))

    @staticmethod
    def rotate(image: np.ndarray, angle: int):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
        image = image.reshape((28,28))
        image = ndimage.rotate(image, angle, reshape=False)
        image = image.reshape(784)
        return image

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        image = image.reshape((28,28))
        image = np.pad(image,((0,dx),(0,dy)))[-28:,-28:]
        image = image.reshape(784)
        return image
    
    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the 
        range [-noise, noise] and added to it. 

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''
        noise = np.random.uniform(low=-noise, high=noise, size=(image.shape))
        image += noise  # In-Place
        image[image > 1] = 1.0
        image[image < 0] = 0.0
        return image

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        
        image = image.reshape((28,28))
        image_ = np.zeros_like(image)
        for row in range(28):
            for col in range(28):
                if int(col+row*tilt) < 28:
                    image_[row,col] = image[row,int(col+row*tilt)]
        image_ = image_.reshape((784))
        return image_

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        noise_bound = np.random.uniform(0.0, 0.2)  # up to 20% noise
        image = self.add_noise(image, noise_bound)
        angle = np.random.randint(-30,30)  # up to +-30 degrees - we don't want to rotate too much to avoid loosing the semantic meaning (a.k.a "sky is up")
        image = self.rotate(image, angle)
        tilt = np.random.uniform(-0.1, 0.1)
        image = self.skew(image, tilt)
        shift = np.random.randint(0,1)  # up to 3 pixels shift - the entire image is 28x28, we don't want to loose all information
        image = self.shift(image, shift, shift)
        
        
        return image

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        while True:
            next_job = self.jobs.get()
            if next_job is None:  # Poison pill means shutdown
                self.jobs.task_done()
                break
            batch_id = next_job['batch ID']
            images_id = next_job['indices ID']
            for i, image_id in enumerate(images_id):
                next_image = self.training_data[0][image_id]
                augmented_image = self.process_image(next_image)
                next_label = self.training_data[1][image_id]
                self.buffer[0][i] = augmented_image
                self.buffer[1][i] = next_label
            self.jobs.task_done()
            self.result.put(self.buffer)
