import numpy as np

class SequentialGraph(object):
    """docstring for SequentialGraph"""
    def __init__(self):
        super(SequentialGraph, self).__init__()
        self.graph = []
        self.compiled = False
    

    def add_layer(self, layer):
        self.graph.append(layer)


    def compile(self, loss_type='cross_entropy', optimizer=None):
        if loss_type == 'cross_entropy':
            loss_layer = Loss(loss_function=loss_type)
            self.graph.append(loss_layer)

        self.compiled = True


    def fit(self, X, y, lr=0.01, epochs=5, batch_size=0):
        if self.compiled:
            for ep in range(epochs):
                y_pred = self.predict(X)
                dZ   = 0
                loss = 0

                for layer in reverse(self.graph):
                    if layer.get_layer_type == 'Loss':
                        loss = layer.forward(y_pred, y)
                        dZ = layer.backward()
                    else:
                        layer.backward(dZ)

                    if layer.trainable:
                        layer.grandient_update(lr)

        return 


    def predict(self, X_test):
        
        output = X_test
        for layer in self.graph:

            if layer.get_layer_type() == 'Loss':
                break
            else:
                output = layer.forward(output)

        return output




