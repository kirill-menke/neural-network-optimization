#pragma once

#include "Tensor.h"

class GPUSgd {
public:
	float learning_rate;
	GPUSgd(float learning_rate):
		learning_rate(learning_rate) {}

	void update(Tensor<float, 4> weights, Tensor<float, 1> bias,
			Tensor<float, 4> gradient_tensor, Tensor<float, 1> gradient_bias);
};

