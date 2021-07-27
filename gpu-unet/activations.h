#pragma once

#include "tensor.h"

class ReLU {
public:
	ReLU() {}

	virtual Tensor<float, 4> forward(const Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(const Tensor<float, 4> &error_tensor);

	Tensor<float, 4> *output_tensor = nullptr;
};

class SoftMax {
public:
	SoftMax() {}

	virtual Tensor<float, 4> forward(const Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(const Tensor<float, 4> &error_tensor);

	Tensor<float, 4> *output_tensor = nullptr;
};

