#pragma once

#include "./tensor.h"

class PerPixelCELoss {
public:
	Tensor<float, 2> forward(const Tensor<float, 4> &prediction, const Tensor<float, 4> &truth);
	Tensor<float, 4> backward(const Tensor<float, 4> &prediction, const Tensor<float, 4> &truth);
};

