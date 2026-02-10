import numpy as np


class Array:
    """
    Custom dataclass to append data to numpy array in a fast manner. 
    """
    def __init__(self, rows, cols=1, dtype=np.float32):
        self.data = np.zeros((rows, cols), dtype=dtype)
        self.rows = rows
        self.cols= cols
        self.dtype = dtype
        self.size = 0

    def add(self, x):
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        if self.cols == 1:
            x = x.reshape(-1,1)
        assert x.shape[1] == self.cols, f"data dims mismatch, got {x.shape[1]} and expected {self.cols}"

        if self.rows-self.size < len(x):
            self.rows += self.rows
            newdata = np.zeros((self.rows,self.cols), dtype=self.dtype)
            newdata[:self.size,:] = self.data[:self.size,:]
            self.data = newdata
            del newdata

        self.data[self.size:self.size+len(x),:] = x
        self.size += len(x)
    
    def getdata(self):
        if self.cols == 1:
            return self.data[:self.size, :].reshape(-1)
        else: 
            return self.data[:self.size, :]