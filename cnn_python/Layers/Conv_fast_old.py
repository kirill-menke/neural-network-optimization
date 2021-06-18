import copy

import numpy as np
from scipy import signal

from Layers.Base import Base

class Conv(Base):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape
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
        # self.input_slice_shape = np.asarray(input_tensor.shape[2:])
        # output_slice_shape = np.ceil(self.input_slice_shape / self.stride_shape).astype(int)
        # feature_map = np.empty((self.num_kernels, len(input_tensor), *output_slice_shape))

        flat_input_tensor = input_tensor.reshape(-1, *input_tensor.shape[2:])
        slicing = (np.s_[self.weights.shape[1] // 2::self.weights.shape[1]], *map(lambda x: np.s_[::x], self.stride_shape))
        
        feature_maps = []
        for kernel, bias in zip(self.weights, self.bias):
            conv = signal.correlate(flat_input_tensor, kernel, mode='same')[slicing]
            feature_maps.append(conv + bias)
    
        return np.array(feature_maps).swapaxes(0, 1)


    def backward(self, error_tensor):
        # Next error tensor: Gradient w.r.t. the different channels of the input
        slicing = (slice(None), slice(None), *map(lambda x: np.s_[::x], self.stride_shape))
        upsampled_error_tensor = np.zeros((*error_tensor.shape[:2], *self.input_tensor.shape[2:]))
        upsampled_error_tensor[slicing] = error_tensor
        flat_error_tensor = upsampled_error_tensor[:, ::-1].reshape(-1, *upsampled_error_tensor.shape[2:])

        kernels = self.weights.swapaxes(0, 1) 
        next_error_tensor = np.empty((len(kernels), len(error_tensor), *self.input_tensor.shape[2:]))
        for k, kernel in enumerate(kernels):
            next_error_tensor[k] = signal.convolve(flat_error_tensor, kernel, mode='same')[kernels.shape[1] // 2::kernels.shape[1]]


        # Gradient w.r.t. weights
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.sum(np.sum(upsampled_error_tensor, axis=(-2, -1)), axis=0)

        npad = ((0, 0), (0, 0), *map(lambda x: (int(np.ceil(0.5 * (x - 1))), int(np.floor(0.5 * (x - 1)))), self.weights.shape[2:]))
        padded_input_tensor = np.pad(self.input_tensor, npad)

        # TODO: Maybe optimize
        for input_sample, error_sample in zip(padded_input_tensor, upsampled_error_tensor):
            for i, error_slice in enumerate(error_sample):
                for j, input_slice in enumerate(input_sample):
                    self.gradient_weights[i][j] += signal.correlate(input_slice, error_slice, mode='valid') 
        
        
        if all(self.optimizer):
            opt_w, opt_b = self.optimizer
            self.weights = opt_w.calculate_update(self.weights, self.gradient_weights)
            self.bias = opt_b.calculate_update(self.bias, self.gradient_bias)

        
        return next_error_tensor.swapaxes(0, 1)
        


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape), self.num_kernels * np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.num_kernels, self.num_kernels, 1) # May be wrong