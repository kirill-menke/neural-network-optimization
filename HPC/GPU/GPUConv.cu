#include <assert.h>

#include "./GPUConv.h"
#include "./cuda-utils.h"

/*
 * Call this kernel per output pixel X/Y and batch.
 */
__global__
static void add_padding(Tensor<float, 4> input, Tensor<float, 4> output, int px, int py) {
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
 * Optimizations:
 * - TODO: Copy filters into shared memory
 * - TODO: Not sure: Copy input into shared memory?
 * - TODO: Less coomplicated loops/filters access
 *
 * Call this kernel per output pixel X/Y and batch.
 */
template<bool flipped>
__global__
static void convolution(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> filters) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = input.dim(0), inputChannels = input.dim(1),
	    width = input.dim(2), height = input.dim(3),
	    outputChannels = output.dim(1);

	int filterWidth = filters.dim(2), filterHeight = filters.dim(3);
	int fw = filterWidth / 2;
	int fh = filterHeight / 2;

	x += fw;
	y += fh;
	if (b > batchSize || x > output.dim(2) || y > output.dim(3))
		return;

	for (int cout = 0; cout < outputChannels; cout++) {
		float value = 0;

		for (int cin = 0; cin < inputChannels; cin++) {
			for (int i = -fw; i <= fw; i++) {
				for (int j = -fh; j <= fh; j++) {
					if (flipped)
						value += input(b, cin, x + i, y + j) * filters.flipped(cin, cout, i + fw, j + fh);
					else
						value += input(b, cin, x + i, y + j) * filters(cout, cin, i + fw, j + fh);
				}
			}
		}

		output(b, cout, x - fw, y - fh) = value;
	}
}

Tensor<float, 4> GPUConv::forward(Tensor<float, 4> &input_tensor) {
	int batchSize = input_tensor.dim(0);

	assert(input_tensor.dim(1) == inputChannels);
	assert(input_tensor.dim(2) == inputWidth);
	assert(input_tensor.dim(3) == inputHeight);

	Tensor<float, 4> padded_input(Tensor<float, 4>::ON_GPU, {
		batchSize,
		inputChannels,
		inputWidth + 2 * (filterWidth / 2),
		inputHeight + 2 * (filterHeight / 2)
	});

	{
		dim3 gridDim = getGridDim(padded_input.dim(2), padded_input.dim(3), batchSize);
		dim3 blockDim = getBlockDim(padded_input.dim(2), padded_input.dim(3), batchSize);
		add_padding<<<gridDim, blockDim>>>(input_tensor, padded_input, filterWidth / 2, filterHeight / 2);
	}

	Tensor<float, 4> output_tensor(Tensor<float, 4>::ON_GPU, {
		batchSize,
		outputChannels,
		inputWidth,
		inputHeight
	});

	{
		dim3 gridDim = getGridDim(inputWidth, inputHeight, batchSize);
		dim3 blockDim = getBlockDim(inputWidth, inputHeight, batchSize);
		convolution<false><<<gridDim, blockDim>>>(padded_input, output_tensor, filters);
	}

	return output_tensor;
}

Tensor<float, 4> GPUConv::backward(Tensor<float, 4> &error_tensor) {
	assert("TODO" && 0);
	return error_tensor;
}
