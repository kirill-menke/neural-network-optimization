import numpy as np

from Layers.Base import Base

class ExpandDims(Base):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        return input_tensor[:, :, np.newaxis, np.newaxis]

    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)
