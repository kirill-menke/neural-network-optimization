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
		int filterHeight,
		int strideX,
		int strideY
	):
		inputChannels(inputChannels),
		imageWidth(imageWidth),
		imageHeight(imageHeight),
		outputChannels(outputChannels),
		filterWidth(filterWidth),
		filterHeight(filterHeight),
		strideX(strideX),
		strideY(strideY),
		weights(Tensor<float, 4>::ON_GPU, { outputChannels, inputChannels, filterWidth, filterHeight }),
		bias(Tensor<float, 1>::ON_GPU, { outputChannels }) {}

	virtual Tensor<float, 4> forward(Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(Tensor<float, 4> &error_tensor);

	int inputChannels;
	int imageWidth;
	int imageHeight;
	int outputChannels;
	int filterWidth;
	int filterHeight;
	int strideX;
	int strideY;
	Tensor<float, 4> weights;
	Tensor<float, 1> bias;

private:
	Tensor<float, 4> *padded_input = nullptr;
};

