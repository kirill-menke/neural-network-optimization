#pragma once

#include "./tensor.h"
#include "./optimizer.h"

class ConvSoftMax {
public:
	ConvSoftMax(
		int input_channels,
		int output_channels):
		input_channels(input_channels),
		output_channels(output_channels),
		weights({ output_channels, input_channels, 1, 1 }),
		bias({ output_channels }) {}

	~ConvSoftMax() {
		if (optimizer != nullptr)
			delete optimizer;

		if (input != nullptr)
			delete input;

		if (output != nullptr)
			delete output;
	}

	virtual Tensor<float, 4> forward(const Tensor<float, 4> &input_tensor);

	/*
	 * Warning: backward() changes values in the error_tensor!
	 */
	virtual Tensor<float, 4> backward(const Tensor<float, 4> &error_tensor);

	int input_channels;
	int output_channels;
	Tensor<float, 4> weights;
	Tensor<float, 1> bias;

	Optimizer *optimizer = nullptr;
	Tensor<float, 4> *input = nullptr;
	Tensor<float, 4> *output = nullptr;
};


