#pragma once

#include <stdint.h>
#include <assert.h>

// #include "../Layer.h"
// #include "unsupported/Eigen/CXX11/Tensor"

#include "Tensor.h"

class GPUMaxPool {
public:
	GPUMaxPool(
		int batch_size,
		int channels,
		int input_width,
		int input_height,
		int stride,
		int pool_size
	):
		batch_size(batch_size),
		channels(channels),
		input_width(input_width),
		input_height(input_height),
		stride(stride),
		pool_size(pool_size),
		maximas(Tensor<uint8_t, 5>::ON_GPU, { batch_size, channels, outputWidth(), outputHeight(), 2 }) {
		assert(stride >= pool_size && "overlapping pools is not supported");
	}

	int batch_size;
	int channels;
	int input_width;
	int input_height;
	int stride;
	int pool_size;

	int outputWidth() const {
		return (input_width - (input_width % pool_size)) / stride;
	}

	int outputHeight() const {
		return (input_height - (input_height % pool_size)) / stride;
	}

	virtual Tensor<float, 4> forward(Tensor<float, 4> &input_tensor);
	virtual Tensor<float, 4> backward(Tensor<float, 4> &error_tensor);

private:
	Tensor<uint8_t, 5> maximas;
};

