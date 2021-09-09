#include "activations.h"
#include "tensor.h"

#include <cassert>

__global__
static void relu_forward(Tensor<float, 4> input, Tensor<float, 4> output)
{
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = input.dim(0), channels = input.dim(1), width = input.dim(2), height = input.dim(3);
	if (c >= channels || x >= width || y >= height)
		return;

	for (int b = 0; b < batch_size; b++) {
		output(b, c, x, y) = fmaxf(0., input(b, c, x, y));
	}
}

__global__
static void relu_backward(Tensor<float, 4> error, Tensor<float, 4> output, Tensor<float, 4> next_error)
{
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = error.dim(0), channels = error.dim(1), width = error.dim(2), height = error.dim(3);
	if (c >= channels || x >= width || y >= height)
		return;

	for (int b = 0; b < batch_size; b++) {
		bool relevant = output(b, c, x, y) != 0.;
		next_error(b, c, x, y) = relevant ? error(b, c, x, y) : 0.;
	}
}

Tensor<float, 4> ReLU::forward(const Tensor<float, 4> &input) {
	output_tensor = new Tensor<float, 4>(input.dim(0), input.dim(1), input.dim(2), input.dim(3));

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, input.dim(1), input.dim(2), input.dim(3));
	relu_forward<<<gridDim, blockDim>>>(input, *output_tensor);

	return *output_tensor;
}

Tensor<float, 4> ReLU::backward(const Tensor<float, 4> &error) {
	Tensor<float, 4> next_error(error.dim(0), error.dim(1), error.dim(2), error.dim(3));

	assert(output_tensor->dim(0) == error.dim(0)
		&& output_tensor->dim(1) == error.dim(1)
		&& output_tensor->dim(2) == error.dim(2)
		&& output_tensor->dim(3) == error.dim(3));

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, error.dim(1), error.dim(2), error.dim(3));
	relu_backward<<<gridDim, blockDim>>>(error, *(this->output_tensor), next_error);

	delete output_tensor;
	output_tensor = nullptr;
	return next_error;
}

__global__
static void softmax_forward(Tensor<float, 4> input, Tensor<float, 4> output)
{
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = input.dim(0), channels = input.dim(1), width = input.dim(2), height = input.dim(3);
	if (b >= batch_size || x >= width || y >= height)
		return;

	float max = input(b, 0, x, y);
	for (int c = 1; c < channels; c++)
		max = fmaxf(max, input(b, c, x, y));

#if 0
	/* `float buf[channels]` would be placed in local memory which is slow. */
	/* writing `output` twice is not optimal. */
	assert(channels == 2);
	float buf[2];
	float sum = 0.;
	for (int c = 0; c < channels; c++) {
		float val = expf(input(b, c, x, y) - max);
		buf[c] = val;
		sum += val;
	}

	for (int c = 0; c < channels; c++)
		output(b, c, x, y) = buf[c] / sum;
#else
	float sum = 0.;
	for (int c = 0; c < channels; c++) {
		float val = expf(input(b, c, x, y) - max);
		output(b, c, x, y) = val;
		sum += val;
	}

	for (int c = 0; c < channels; c++)
		output(b, c, x, y) /= sum;
#endif
}

__global__
static void softmax_backward(Tensor<float, 4> error, Tensor<float, 4> output, Tensor<float, 4> next_error) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = error.dim(0), channels = error.dim(1), width = error.dim(2), height = error.dim(3);
	if (b >= batch_size || x >= width || y >= height)
		return;

	float sum = 0.;
	for (int c = 0; c < channels; c++)
		sum += error(b, c, x, y) * output(b, c, x, y);

	for (int c = 0; c < channels; c++)
		next_error(b, c, x, y) = output(b, c, x, y) * (error(b, c, x, y) - sum);
}

Tensor<float, 4> SoftMax::forward(const Tensor<float, 4> &input) {
	output_tensor = new Tensor<float, 4>(input.dim(0), input.dim(1), input.dim(2), input.dim(3));

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, input.dim(0), input.dim(2), input.dim(3));
	softmax_forward<<<gridDim, blockDim>>>(input, *output_tensor);

	return *output_tensor;
}

Tensor<float, 4> SoftMax::backward(const Tensor<float, 4> &error) {
	Tensor<float, 4> next_error(error.dim(0), error.dim(1), error.dim(2), error.dim(3));

	assert(output_tensor->dim(0) == error.dim(0)
		&& output_tensor->dim(1) == error.dim(1)
		&& output_tensor->dim(2) == error.dim(2)
		&& output_tensor->dim(3) == error.dim(3));

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, error.dim(0), error.dim(2), error.dim(3));
	softmax_backward<<<gridDim, blockDim>>>(error, *output_tensor, next_error);

	delete output_tensor;
	output_tensor = nullptr;
	return next_error;
}

