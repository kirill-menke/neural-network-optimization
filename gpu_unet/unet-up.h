#pragma once

#include <cassert>

#include "./tensor.h"

/*
 * Forward case:
 * output = concat(input1, Upsample.forward(input2))
 *
 * Backward case:
 * output_err = Upsample.backward(error[:, skipcon_channels..(error.dim(1)), ;, ;])
 */
class UnetUp {
public:
	UnetUp(int pool_size, int skipcon_channels, int upsample_channels):
		pool_size(pool_size),
		skipcon_channels(skipcon_channels),
		upsample_channels(upsample_channels) {}

	int pool_size;
	int skipcon_channels;
	int upsample_channels;

	/*
	 * input1: Skip connections, take them as they are and merge them with upsampled input2.
	 * input2: From the layer below this one, upsample and merge with input1.
	 */
	virtual Tensor<float, 4> forward(const Tensor<float, 4> &input1, const Tensor<float, 4> &input2);

	/*
	 * Take only the channels really needed from the error_tensor and downsample.
	 */
	virtual Tensor<float, 4> backward(const Tensor<float, 4> &error);
};

