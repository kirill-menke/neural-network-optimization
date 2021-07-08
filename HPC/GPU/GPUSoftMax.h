#pragma once

#include "Tensor.h"

class GPUSoftMax {
public:
	GPUSoftMax() {}

	virtual Tensor<float, 4> forward(Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(Tensor<float, 4> &error_tensor);

private:
	Tensor<float, 4> *output_tensor = nullptr;
};

