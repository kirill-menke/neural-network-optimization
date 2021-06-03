#pragma once

// #include "../Layer.h"
// #include "unsupported/Eigen/CXX11/Tensor"

#include "Tensor.h"

class GPUConv {
public:
	GPUConv(
		int inputChannels,
		int inputWidth,
		int inputHeight,
		int outputChannels,
		int filterWidth,
		int filterHeight
	):
		inputChannels(inputChannels),
		inputWidth(inputWidth),
		inputHeight(inputHeight),
		outputChannels(outputChannels),
		filterWidth(filterWidth),
		filterHeight(filterHeight),
		filters(Tensor<float, 4>::ON_GPU, { outputChannels, inputChannels, filterWidth, filterHeight }) {}

	virtual Tensor<float, 4> forward(Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(Tensor<float, 4> &error_tensor);

	int inputChannels;
	int inputWidth;
	int inputHeight;
	int outputChannels;
	int filterWidth;
	int filterHeight;
	Tensor<float, 4> filters;
};

