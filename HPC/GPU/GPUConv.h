#pragma once

// #include "../Layer.h"
// #include "unsupported/Eigen/CXX11/Tensor"

#include "Tensor.h"

class GPUConv {
public:
	GPUConv(
		int inputChannels,
		int imageWidth,
		int imageHeight,
		int outputChannels,
		int filterWidth,
		int filterHeight
	):
		inputChannels(inputChannels),
		imageWidth(imageWidth),
		imageHeight(imageHeight),
		outputChannels(outputChannels),
		filterWidth(filterWidth),
		filterHeight(filterHeight),
		filters(Tensor<float, 4>::ON_GPU, { outputChannels, inputChannels, filterWidth, filterHeight }) {}

	virtual Tensor<float, 4> forward(Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(Tensor<float, 4> &error_tensor);

	int inputChannels;
	int imageWidth;
	int imageHeight;
	int outputChannels;
	int filterWidth;
	int filterHeight;
	Tensor<float, 4> filters;

private:
	Tensor<float, 4> *padded_input = nullptr;
};

