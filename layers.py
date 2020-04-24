import numpy as np
from gates import SumGate, MultiplyGate, ReluGate, SigmoidGate, SoftmaxGate, CrossEntropyGate
from 

class Dense(object):
    layer_type = 'Dense'
    trainable  = True

    def __init__(self, input_size, output_size, activation='relu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation


        # Weight Initialization
        self.W = np.random.randn(self.output_size, self.input_size)
        self.B = np.random.randn(self.output_size, 1)

    def forward(self, X)
        self.m_gate = MultiplyGate()
        self.s_gate  = SumGate()

        if self.activation == 'relu':
            self.a_gate = ReluGate()
        if self.activation == 'sigmoid':
            self.a_gate = SigmoidGate()
        if self.activation == 'softmax':
            self.a_gate = SoftmaxGate()

        m_out = self.m_gate.forward(self.W, X)
        s_out = self.s_gate.forward(m_out, self.B)
        a_out = self.a_gate.forward(s_out)

        return a_out        

    def backward(self, dZ):

        dA = self.a_gate.backward(dZ)
        dWX, self.dB = self.s_gate.backward(dA)
        self.dW, dX = self.m_gate.backward(dWX)

        return self.dW

    def gradient_update(self, lr):
        self.W -= lr * self.dW
        self.B -= lr * self.dB 

    def get_params(self):
        return self.W, self.B

    def get_layer_type(self):
        return self.layer_type


class Activation(object):
    layer_type = 'Activation'
    trainable = False

    def __init__(self, activation='relu'):
        self.activation = activation
        
        if self.activation == 'relu':
            self.a_gate = ReluGate()
        if self.activation == 'sigmoid':
            self.a_gate = SigmoidGate()
        if self.activation == 'softmax':
            self.a_gate = SoftmaxGate()

    def forward(self, X):
        return self.a_gate.forward(X)

    def backward(self, dZ):
        return self.a_gate.backward(dZ)


class Loss(object):
    layer_type = 'Loss'
    trainable  = False

    def __init__(self, loss_function='cross_entropy'):
        if loss_function == 'cross_entropy':
            self.loss_gate = CrossEntropyGate()

    def forward(self, pred, truth):
        return self.loss_gate.forward(pred, truth)

    def backward(self):
        return self.loss_gate.backward()

    def get_layer_type(self):
        return self.layer_type

