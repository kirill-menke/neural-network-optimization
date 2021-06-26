#include <assert.h>

#include "./GPUConv.h"
#include "./cuda-utils.h"

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
 * Upsample (used for error_tensor in backward)
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

/*
 * Calculate gradient_weights:
 *
 * Call this kernel per filter pixel X/Y and batch (TODO: Could also include filter here for perf.)
 * TODO: Can this be represented as a wired convolution as well?
 */
__global__
void convBackwardGradientWeights(Tensor<float, 4> error_tensor, Tensor<float, 4> input_tensor, Tensor<float, 4> gradient_weights, int strideX, int strideY) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int filterWidth = gradient_weights.dim(2), filterHeight = gradient_weights.dim(3);
	int batchSize = input_tensor.dim(0);
	if (i >= filterWidth || j >= filterHeight || b >= batchSize)
		return;

	int imageWidth = error_tensor.dim(2), imageHeight = error_tensor.dim(3);
	int inputChannels = input_tensor.dim(1), outputChannels = error_tensor.dim(1);
	for (int f = 0; f < outputChannels; f++) {
		for (int c = 0; c < inputChannels; c++) {
			float err = 0.0;

			for (int x = 0; x < imageWidth; x += strideX) {
				for (int y = 0; y < imageHeight; y += strideY) {
					err += input_tensor(b, c, x + i, y + j) * error_tensor(b, f, x, y);
				}
			}

			gradient_weights(f, c, i, j) = err;
		}
	}
}

/*
 * TODO: Do some sort of parallel reduction instead?
 */
__global__
void convBackwardGradientBias(Tensor<float, 4> error_tensor, Tensor<float, 1> gradient_bias) {
	int f = threadIdx.x + blockIdx.x * blockDim.x;
	if (f >= gradient_bias.dim(0))
		return;

	int batchSize = error_tensor.dim(0),
	    width = error_tensor.dim(2),
	    height = error_tensor.dim(3);

	float value = 0.;
	for (int b = 0; b < batchSize; b++) {
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				value += error_tensor(b, f, x, y);
			}
		}
	}
	gradient_bias(f) = value;
}

Tensor<float, 4> GPUConv::forward(Tensor<float, 4> &input_tensor) {
	int batchSize = input_tensor.dim(0);

	assert(input_tensor.dim(1) == inputChannels);
	assert(input_tensor.dim(2) == imageWidth);
	assert(input_tensor.dim(3) == imageHeight);

	this->padded_input = new Tensor<float, 4>(Tensor<float, 4>::ON_GPU, {
		batchSize,
		inputChannels,
		imageWidth + 2 * (filterWidth / 2),
		imageHeight + 2 * (filterHeight / 2)
	});

	{
		dim3 gridDim = getGridDim(padded_input->dim(2), padded_input->dim(3), batchSize);
		dim3 blockDim = getBlockDim(padded_input->dim(2), padded_input->dim(3), batchSize);
		addPadding<<<gridDim, blockDim>>>(input_tensor, *padded_input, filterWidth / 2, filterHeight / 2);
	}

	int outputWidth = imageWidth / strideX, outputHeight = imageHeight / strideY;

	Tensor<float, 4> output_tensor(Tensor<float, 4>::ON_GPU, {
		batchSize,
		outputChannels,
		outputWidth,
		outputHeight
	});

	{
		dim3 gridDim = getGridDim(outputWidth, outputHeight, batchSize);
		dim3 blockDim = getBlockDim(outputWidth, outputHeight, batchSize);
		convolution<false><<<gridDim, blockDim>>>(*padded_input, output_tensor, weights, bias, strideX, strideY);
	}

	return output_tensor;
}

/*
 * Opt./Simplify for stride = 1?
 */
Tensor<float, 4> GPUConv::backward(Tensor<float, 4> &error_tensor) {
	int batchSize = error_tensor.dim(0);
	int outputWidth = imageWidth / strideX, outputHeight = imageHeight / strideY;
	assert(error_tensor.dim(1) == inputChannels);
	assert(error_tensor.dim(2) == outputWidth);
	assert(error_tensor.dim(3) == outputHeight);
	assert(padded_input->dim(0) == batchSize);

	Tensor<float, 4> upsampled_error_tensor(Tensor<float, 4>::ON_GPU, {
		batchSize,
		outputChannels,
		imageWidth,
		imageHeight
	});

	{
		dim3 gridDim = getGridDim(imageWidth, imageHeight, batchSize);
		dim3 blockDim = getBlockDim(imageWidth, imageHeight, batchSize);
		upsample<<<gridDim, blockDim>>>(error_tensor, upsampled_error_tensor, strideX, strideY);
	}

	Tensor<float, 4> padded_error_tensor(Tensor<float, 4>::ON_GPU, {
		batchSize,
		outputChannels,
		imageWidth + 2 * (filterWidth / 2),
		imageHeight + 2 * (filterHeight / 2)
	});

	{
		dim3 gridDim = getGridDim(padded_error_tensor.dim(2), padded_error_tensor.dim(3), batchSize);
		dim3 blockDim = getBlockDim(padded_error_tensor.dim(2), padded_error_tensor.dim(3), batchSize);
		addPadding<<<gridDim, blockDim>>>(upsampled_error_tensor, padded_error_tensor, filterWidth / 2, filterHeight / 2);
	}


	Tensor<float, 4> next_error_tensor(Tensor<float, 4>::ON_GPU, {
		batchSize,
		inputChannels,
		imageWidth,
		imageHeight
	});

	{
		dim3 gridDim = getGridDim(imageWidth, imageHeight, batchSize);
		dim3 blockDim = getBlockDim(imageWidth, imageHeight, batchSize);
		convolution<true><<<gridDim, blockDim>>>(padded_error_tensor, next_error_tensor, weights, bias, 1, 1);
	}

	Tensor<float, 4> gradient_weights(Tensor<float, 4>::ON_GPU, {
		outputChannels,
		inputChannels,
		filterWidth,
		filterHeight
	});

	{
		dim3 gridDim = getGridDim(filterWidth, filterHeight, batchSize);
		dim3 blockDim = getBlockDim(filterWidth, filterHeight, batchSize);
		convBackwardGradientWeights<<<gridDim, blockDim>>>(upsampled_error_tensor, *padded_input, gradient_weights, strideX, strideY);
	}

	Tensor<float, 1> gradient_bias(Tensor<float, 1>::ON_GPU, { outputChannels });

	{
		dim3 gridDim = getGridDim(outputChannels, 1, 1);
		dim3 blockDim = getBlockDim(outputChannels, 1, 1);
		convBackwardGradientBias<<<gridDim, blockDim>>>(error_tensor, gradient_bias);
	}

	// TODO: Pass gradient_weights/bias on to Optimizer or something like that!
	gradient_weights.destroy();
	gradient_bias.destroy();

	padded_error_tensor.destroy();
	upsampled_error_tensor.destroy();

	this->padded_input->destroy();
	delete this->padded_input;
	this->padded_input = nullptr;
	return next_error_tensor;
}
