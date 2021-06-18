import copy

import numpy as np
from scipy import signal

from Layers.Base import Base

class Conv(Base):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.stride_shape = np.asarray(stride_shape)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.rand(num_kernels, *convolution_shape)
        self.bias = np.random.rand(num_kernels)

        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer_weights = None
        self._optimizer_bias = None

    
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer_weights, self._optimizer_bias

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer_weights = value
        self._optimizer_bias = copy.deepcopy(value)

    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.input_slice_shape = np.asarray(input_tensor[0][0].shape)
        output_slice_shape = np.ceil(self.input_slice_shape / self.stride_shape).astype(int)
        feature_map = np.empty((input_tensor.shape[0], self.num_kernels, *output_slice_shape))

        for b, input_sample in enumerate(input_tensor):
            slicing = (input_sample.shape[0] // 2, *map(lambda x: np.s_[::x], self.stride_shape))
            
            for k, kernel in enumerate(self.weights):
                feature_map[b][k] = signal.correlate(input_sample, kernel, mode='same')[slicing]
                feature_map[b][k] += self.bias[k]
            
        return feature_map


    def backward(self, error_tensor):

        # Next error tensor: Gradient w.r.t. the different channels of input
        kernels = self.weights.swapaxes(0, 1) 
        next_error_tensor = np.empty((error_tensor.shape[0], kernels.shape[0], *self.input_slice_shape))
        
        for b, error_sample in enumerate(error_tensor):
            u_error_sample = np.zeros((kernels.shape[1], *self.input_slice_shape))
            slicing = (slice(None), *map(lambda x: np.s_[::x], self.stride_shape))
            u_error_sample[slicing] = error_sample[::-1]

            for k, kernel in enumerate(kernels):
                next_error_tensor[b, k] = signal.convolve(u_error_sample, kernel, mode='same')[len(error_sample) // 2]
        

        # Gradient w.r.t. weights
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

        for input_sample, error_sample in zip(self.input_tensor, error_tensor):
            u_error_sample = np.zeros((len(self.weights), *self.input_slice_shape))
            idx = (slice(None), *map(lambda x: np.s_[::x], self.stride_shape))
            u_error_sample[idx] = error_sample

            sample_gradients = np.empty((len(self.weights), len(input_sample), *self.weights.shape[2:]))
            for i, (error_slice, kernel) in enumerate(zip(u_error_sample, self.weights)):
                self.gradient_bias[i] += np.sum(error_slice)
                
                for j, input_slice in enumerate(input_sample):
                    pad = (*map(lambda x: (int(np.ceil(0.5 * (x - 1))), int(np.floor(0.5 * (x - 1)))), kernel.shape[1:]), )
                    p_input_slice = np.pad(input_slice, pad)

                    self.gradient_weights[i][j] += signal.correlate(p_input_slice, error_slice, mode='valid') 
                    
        
        if all(self.optimizer):
            opt_w, opt_b = self.optimizer
            self.weights = opt_w.calculate_update(self.weights, self.gradient_weights)
            self.bias = opt_b.calculate_update(self.bias, self.gradient_bias)

        
        return next_error_tensor
        


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape), self.num_kernels * np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.num_kernels, self.num_kernels, 1) # May be wrong