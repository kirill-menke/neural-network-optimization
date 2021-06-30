#pragma once

// #include "../Layer.h"
// #include "unsupported/Eigen/CXX11/Tensor"

#include "Tensor.h"
#include "GPUOptimizer.h"

class GPUConv {
public:
	GPUConv(
		int inputChannels,
		int outputChannels,
		int imageWidth,
		int imageHeight,
		int filterWidth,
		int filterHeight,
		int strideX,
		int strideY
	):
		inputChannels(inputChannels),
		outputChannels(outputChannels),
		imageWidth(imageWidth),
		imageHeight(imageHeight),
		filterWidth(filterWidth),
		filterHeight(filterHeight),
		strideX(strideX),
		strideY(strideY),
		weights({ outputChannels, inputChannels, filterWidth, filterHeight }),
		bias({ outputChannels }) {}

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

	GPUSgd *optimizer = nullptr;

private:
	Tensor<float, 4> *padded_input = nullptr;
};

