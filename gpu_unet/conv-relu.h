#pragma once

#include "./tensor.h"
#include "./optimizer.h"

class ConvReLU {
public:
	ConvReLU(int input_channels, int output_channels, int filter_size) :
		input_channels(input_channels),
		output_channels(output_channels),
		filter_size(filter_size),
		weights({ output_channels, input_channels, filter_size, filter_size }),
		bias({ output_channels }) {}

	~ConvReLU() {
		if (optimizer != nullptr)
			delete optimizer;

		if (input != nullptr)
			delete input;

		if (output != nullptr)
			delete output;
	}

	virtual Tensor<float, 4> forward(const Tensor<float, 4>& input_tensor);
	virtual Tensor<float, 4> backward(const Tensor<float, 4>& error_tensor);

	int input_channels;
	int output_channels;
	int filter_size;
	Tensor<float, 4> weights;
	Tensor<float, 1> bias;

	Optimizer* optimizer = nullptr;
	Tensor<float, 4>* input = nullptr;
	Tensor<float, 4>* output = nullptr;
};