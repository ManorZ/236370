
import os
from multiprocessing import Pipe, Lock, Value

# counter = 0

class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self.parent_conn, self.child_conn = Pipe()
        self.lock = Lock()
        # global counter
        # self.counter = 0
        self.counter = Value('i', 0)  # Shared Memory based counter.

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self.lock.acquire()
        try:    
            # print(os.getppid(), os.getpid(), id(self.counter), self.counter.value)
            self.child_conn.send(msg)
            # self.counter += 1
            self.counter.value += 1
        finally:
            self.lock.release()
        
        
    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        msg = self.parent_conn.recv()
        # print(os.getpid(), id(self.counter), self.counter.value)
        # self.counter -= 1
        # with self.counter.get_lock():
        self.counter.value -= 1
        return msg
    
    def length(self):
        '''Get the number of messages currently in the queue
            
        Return
        ------
        An integer
        '''
        # return self.counter
        return self.counter.value