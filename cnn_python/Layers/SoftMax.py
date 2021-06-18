import numpy as np

from Layers.Base import Base

class SoftMax(Base):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        norm_input = input_tensor - np.amax(input_tensor, axis=1, keepdims=True)
        exp_input = np.exp(norm_input)
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output

    def backward(self, error_tensor):
        sum_ = np.sum(error_tensor * self.output, axis=1, keepdims=True)
        return self.output * (error_tensor - sum_)