import numpy as np

from Layers.Base import Base

class Flatten(Base):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        return input_tensor.reshape(self.shape[0], -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)
