#pragma once
#include "Tensor.h"
#include "GPUOptimizer.h"

class GPUTransConv {
public:
	GPUTransConv(
		int inputChannels,
		int outputChannels,
		int imageWidth,
		int imageHeight,
		int filterWidth,
		int filterHeight,
		int strideX,
		int strideY
	) :
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

	/* Assumes the associated convolution is described by 
	 * padding = 0 and input_size - kernel_size = multiple of stride
	 * 
	 * https://arxiv.org/pdf/1603.07285.pdf (Relationship 12)
	 */
	virtual Tensor<float, 4> forward(Tensor<float, 4>& input_tensor);
	virtual Tensor<float, 4> backward(Tensor<float, 4>& error_tensor);

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

	GPUSgd* optimizer = nullptr;

private:
	Tensor<float, 4>* padded_input = nullptr;
};