#pragma once

#include <cassert>

#include "./tensor.h"

/*
 * Forward Case:
 * output = MaxPool.forward(input)
 *
 * Backward Case:
 * output_err = MaxPool.backward(error1) + error2[:, 0..error1.dim(1), :, :];
 */
class UnetDown {
public:
	UnetDown(int pool_size):
		pool_size(pool_size) {}

	int pool_size;

	virtual Tensor<float, 4> forward(const Tensor<float, 4> &input);

	/*
	 * error1: smaller error_tensor coming from "below" this layer.
	 * error2: bigger error_tensor coming from the skip connections.
	 *
	 * error2 will have more channels than this layer needs!
	 * Take the channels in range `[0..channels]` here.
	 */
	virtual Tensor<float, 4> backward(const Tensor<float, 4> &error1, const Tensor<float, 4> &error2);

private:
	Tensor<uint8_t, 5> *maximas = nullptr;
};

