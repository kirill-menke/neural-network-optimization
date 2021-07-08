#pragma once
#include "./Tensor.h"
#include "./cuda-utils.h"
#include "device_launch_parameters.h"


/*
 * Call this kernel per output pixel X/Y and batch.
 */
__global__
static void addPadding(Tensor<float, 4> input, Tensor<float, 4> output, int px, int py) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = input.dim(0), channels = input.dim(1),
		inputWidth = input.dim(2), inputHeight = input.dim(3),
		outputWidth = output.dim(2), outputHeight = output.dim(3);

	if (x >= outputWidth || y >= outputHeight)
		return;

	for (int c = 0; c < channels; c++) {
		if (x < px || y < py || x >= px + inputWidth || y >= py + inputHeight)
			output(b, c, x, y) = 0.0;
		else
			output(b, c, x, y) = input(b, c, x - px, y - py);
	}
}

/*
 * Upsample using zeros (used for error_tensor in backward)
 *
 * Call this kernel per output pixel X/Y and batch.
 */
__global__
static void upsample(Tensor<float, 4> input, Tensor<float, 4> output, int strideX, int strideY) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = input.dim(0), channels = input.dim(1),
		inputWidth = input.dim(2), inputHeight = input.dim(3),
		outputWidth = output.dim(2), outputHeight = output.dim(3);

	if (x >= outputWidth || y >= outputHeight || b >= batchSize)
		return;

	for (int c = 0; c < channels; c++) {
		if (x % strideX == 0 && y % strideY == 0 && x / strideX < inputWidth && y / strideY < inputHeight)
			output(b, c, x, y) = input(b, c, x / strideX, y / strideY);
		else
			output(b, c, x, y) = 0.0;
	}
}


/*
 * Upsample by replicating input (used for input_tensor in forward)
 *
 * Call this kernel per output pixel X/Y and batch.
 */
__global__
static void upsample_neighbor(Tensor<float, 4> input, Tensor<float, 4> output, int strideX, int strideY) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = input.dim(0), channels = input.dim(1),
		inputWidth = input.dim(2), inputHeight = input.dim(3),
		outputWidth = output.dim(2), outputHeight = output.dim(3);

	if (x >= outputWidth || y >= outputHeight || b >= batchSize)
		return;

	for (int c = 0; c < channels; c++) {
		output(b, c, x, y) = input(b, c, x / strideX, y / strideY);
	}
}


/*
 * Optimizations:
 * - TODO: Copy filters into shared memory
 * - TODO: Not sure: Copy input into shared memory?
 *
 * Call this kernel per output pixel X/Y and batch.
 */
template<bool backward>
__global__
static void convolution(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias, int strideX, int strideY) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = input.dim(0), inputChannels = input.dim(1),
		outputWidth = output.dim(2), outputHeight = output.dim(3),
		outputChannels = output.dim(1);

	int filterWidth = weights.dim(2), filterHeight = weights.dim(3);

	if (b > batchSize || x >= outputWidth || y >= outputHeight)
		return;

	for (int cout = 0; cout < outputChannels; cout++) {
		float value = 0;

		for (int cin = 0; cin < inputChannels; cin++) {
			for (int i = 0; i < filterWidth; i++) {
				for (int j = 0; j < filterHeight; j++) {
					float inputVal = input(b, cin, x * strideX + i, y * strideY + j);
					if (backward)
						value += inputVal * weights.flipped(cin, cout, i, j);
					else
						value += inputVal * weights(cout, cin, i, j);
				}
			}
		}

		if (!backward)
			value += bias(cout);

		output(b, cout, x, y) = value;
	}
}


__global__
static void backwardGradients(Tensor<float, 4> error_tensor, Tensor<float, 4> upsampled_error_tensor,
	Tensor<float, 4> input_tensor,
	Tensor<float, 4> gradient_weights, Tensor<float, 1> gradient_bias, int strideX, int strideY) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int f = threadIdx.z + blockIdx.z * blockDim.z;

	int filterWidth = gradient_weights.dim(2), filterHeight = gradient_weights.dim(3), numFilters = gradient_weights.dim(0);
	if (i >= filterWidth || j >= filterHeight || f >= numFilters)
		return;

	int batchSize = input_tensor.dim(0), inputChannels = input_tensor.dim(1),
		outputWidth = error_tensor.dim(2), outputHeight = error_tensor.dim(3);

	int w = upsampled_error_tensor.dim(2),
		h = upsampled_error_tensor.dim(3);

	// Calc weights:
	for (int c = 0; c < inputChannels; c++) {
		float err = 0.;

		for (int b = 0; b < batchSize; b++)
			for (int x = 0; x < w; x += strideX)
				for (int y = 0; y < h; y += strideY)
					err += input_tensor(b, c, x + i, y + j) * upsampled_error_tensor(b, f, x, y);

		gradient_weights(f, c, i, j) = err;
	}

	// Calc bias:
	if (i != 0 || j != 0)
		return;

	float err = 0.;
	for (int b = 0; b < batchSize; b++)
		for (int x = 0; x < outputWidth; x++)
			for (int y = 0; y < outputHeight; y++)
				err += error_tensor(b, f, x, y);

	gradient_bias(f) = err;
}

/* Standard matrix multiplication */
__global__ 
static void dot_product(Tensor<float, 2> A, Tensor<float, 2> B, Tensor<float, 2> C)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0;
	if (row < A.dim(0) && col < B.dim(1))
	{
		for (int i = 0; i < B.dim(0); i++)
		{
			sum += A(row, i) * B(i, col);
		}
		C(row, col) = sum;
	}
}