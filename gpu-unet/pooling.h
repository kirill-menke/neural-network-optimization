#pragma once

#include <cassert>

#include "tensor.h"

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

