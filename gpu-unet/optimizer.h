#pragma once

#include "tensor.h"

class Optimizer {
public:
	Optimizer(float learning_rate):
		learning_rate(learning_rate) {}

	virtual ~Optimizer() {}

	float learning_rate;
	virtual void update(Tensor<float, 4> &weights, Tensor<float, 1> &bias, Tensor<float, 4> &gradient_weights, Tensor<float, 1> &gradient_bias) = 0;
};

class Sgd: public Optimizer {
public:
	Sgd(float learning_rate):
		Optimizer(learning_rate) {}

	void update(Tensor<float, 4> &weights, Tensor<float, 1> &bias, Tensor<float, 4> &gradient_weights, Tensor<float, 1> &gradient_bias);
};

