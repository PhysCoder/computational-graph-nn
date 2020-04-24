import numpy as np


class Operation(object):

    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class SumGate(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        self.A = A
        self.B = B

        return self.A + self.B

    def backward(self， dZ):
        return dZ, dZ


class MultiplyGate(Operation):
    def __init__(self):
        super().__init()

    def forward(self, A, B):
        self.A = A
        self.B = B

        return np.matmul(self.A, self.B)

    def backward(self, dZ):

        return np.matmul(dZ.T, self.B.T), np.matmul(self.A.T, dZ)


class ReluGate(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        '''
        Input:
            X: d x N matrix with each observation forming a column vector of d x 1
        Return:
            d x N matrix
        '''
        self.X = X
        return np.maximum(self.X, 0.0)

    def backward(self, dZ):
        return np.dZ * (self.X > 0.0)


class SigmoidGate(Operation):
    def __init__(self):
        super().__init__(self)

    def sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)

    def forward(self, X):
        '''
        Input:
            X: d x N matrix with each observation forming a column vector of d x 1
        Return: 
            d x N matrix
        '''
        self.X = X
        self.sigX = self.sigmoid(X)

        return self.sigX

    def backward(self, dZ):

        return dZ * np.diag( (self.sigX * (1 - self.sigX)).diagonal() )


class SoftmaxGate(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        '''
        Input:
            X: C x N matrix of N observations and C classes
        Return:
            C x N matrix
        '''
        self.X = X
        self.exp = np.exp(X)
        self.sftmx = self.exp / np.sum(self.exp, axis=0, keepdims=True)
        
        return self.sftmx 

    def backward(self, dZ):
        dim = sef.sftmx.shape[0]
        driv_mat = self.sftmx.T * (np.eye(dim) - np.vstack([self.sftmx.T]*dim))

        return np.matmul(dZ.T, driv_mat)


class CrossEntropyGate(Operation):
    def __init__(self):
        super().__init__()

    def forward(self, sftmx, y):
        '''
        Input:
            sftmx: C x N matrix of N observations and C classes
            y: C x N matrix of N observations and C classes, one-hot encoded
        Return:
            scalar cross entropy value
        '''
        self.sftmx = sftmx
        self.y = y

        return -np.mean(np.sum(self.y * np.log(sef.sftmx), axis=1))


    def backward(self):

        return self.y / self.sftmx





