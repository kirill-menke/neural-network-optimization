#pragma once

#include "Tensor.h"

class GPUReLU {
public:
	GPUReLU() {}

	virtual Tensor<float, 4> forward(const Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(const Tensor<float, 4> &error_tensor);

private:
	Tensor<float, 4> *output_tensor = nullptr;
};

