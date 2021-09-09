#pragma once

#include <cassert>

#include "tensor.h"
#include "optimizer.h"

class MaxPool {
public:
	MaxPool(int pool_size): pool_size(pool_size) {}

	int pool_size;

	virtual Tensor<float, 4> forward(const Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(const Tensor<float, 4> &error_tensor);

private:
	Tensor<uint8_t, 5> *maximas;
};

class Upsample {
public:
	Upsample(int pool_size): pool_size(pool_size) {}

	int pool_size;

	virtual Tensor<float, 4> forward(const Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(const Tensor<float, 4> &error_tensor);
};


/* Assumes the associated convolution is described by
 * padding = 0 and input_size - kernel_size = multiple of stride
 *
 * https://arxiv.org/pdf/1603.07285.pdf (Relationship 12)
 */
class TransposedConv {
	int inputChannels;
	int outputChannels;
	int filterSize;
	int strideX;
	int strideY;

	Tensor<float, 4> weights;
	Tensor<float, 1> bias;

	Optimizer* optimizer = nullptr;
	Tensor<float, 4>* padded_input = nullptr;

public:
	TransposedConv(int inputChannels, int outputChannels, int filterSize, int strideX, int strideY) :
		inputChannels(inputChannels),
		outputChannels(outputChannels),
		filterSize(filterSize),
		strideX(strideX),
		strideY(strideY),
		weights({ outputChannels, inputChannels, filterSize, filterSize }),
		bias({ outputChannels }) {}

	virtual Tensor<float, 4> forward(Tensor<float, 4>& input_tensor);
	virtual Tensor<float, 4> backward(Tensor<float, 4>& error_tensor);
};

