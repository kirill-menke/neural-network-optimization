import numpy as np
from Layers.Base import Base

class ReLU(Base):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        ret = np.maximum(0, input_tensor)
        return ret
        
    def backward(self, error_tensor):
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor