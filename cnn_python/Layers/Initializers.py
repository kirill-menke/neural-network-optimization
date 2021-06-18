import numpy as np

class Constant:
    def __init__(self, value = 0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.value)



class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.random(weights_shape)


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2 / (fan_out + fan_in))
        return np.random.normal(0, std, weights_shape)


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, weights_shape)