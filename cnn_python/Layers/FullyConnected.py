import numpy as np

from Layers.Base import Base

class FullyConnected(Base):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size + 1
        self.output_size = output_size
        self.weights = np.random.rand(output_size, self.input_size).T
        self._optimizer = None
        self._gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value


    def forward(self, input_tensor):
        self.input_tensor = np.column_stack((input_tensor, np.ones(len(input_tensor))))
        return self.input_tensor @ self.weights

    def backward(self, error_tensor):
        next_error_tensor = np.delete(error_tensor @ self.weights.T, -1, 1)

        self.gradient_weights = self.input_tensor.T @ error_tensor
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return next_error_tensor


    def initialize(self, weights_initializer, bias_initializer):
        self.weights[:-1, :] = weights_initializer.initialize(self.weights.shape, self.input_size - 1, self.output_size)[:-1, :]
        self.weights[-1, :] = bias_initializer.initialize(self.output_size, self.output_size, 1)
