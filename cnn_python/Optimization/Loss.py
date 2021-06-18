import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        losses = -np.log(input_tensor[label_tensor==1] + np.finfo(float).eps)
        return np.sum(losses)

    def backward(self, label_tensor):
        grad = -label_tensor / self.input_tensor
        return grad