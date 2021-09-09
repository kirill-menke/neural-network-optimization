#pragma once

#include "tensor.h"
#include "optimizer.h"

class Conv {
public:
	Conv(
		int input_channels,
		int output_channels,
		int filter_size):
		input_channels(input_channels),
		output_channels(output_channels),
		filter_size(filter_size),
		weights({ output_channels, input_channels, filter_size, filter_size }),
		bias({ output_channels }) {}

	~Conv() {
		if (optimizer != nullptr)
			delete optimizer;

		if (input != nullptr)
			delete input;
	}

	virtual Tensor<float, 4> forward(const Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(const Tensor<float, 4> &error_tensor);

	int input_channels;
	int output_channels;
	int filter_size;
	Tensor<float, 4> weights;
	Tensor<float, 1> bias;

	Optimizer *optimizer = nullptr;
	Tensor<float, 4> *input = nullptr;
};

