#include <limits>
#include <cmath>

#include "GPUSoftMax.h"
#include "cuda-utils.h"

__global__
static void forwardKernel(Tensor<float, 4> input, Tensor<float, 4> output)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = input.dim(0), channels = input.dim(1), width = input.dim(2), height = input.dim(3);
	if (b >= batch_size || x >= width || y >= height)
		return;

	float max = -std::numeric_limits<float>::infinity();
	for (int c = 0; c < channels; c++) {
		float val = input(b, c, x, y);
		if (val > max)
			max = val;
	}

	float sum = 0.;
	for (int c = 0; c < channels; c++) {
		float val = expf(input(b, c, x, y) - max);
		output(b, c, x, y) = val;
		sum += val;
	}

	for (int c = 0; c < channels; c++)
		output(b, c, x, y) /= sum;
}

__global__
static void backwardKernel(Tensor<float, 4> error, Tensor<float, 4> output, Tensor<float, 4> next_error)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
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

Tensor<float, 4> GPUSoftMax::forward(Tensor<float, 4> &input) {
	this->output_tensor = new Tensor<float, 4>({
		input.dim(0), input.dim(1), input.dim(2), input.dim(3) });

	dim3 gridDim = getGridDim(input.dim(2), input.dim(3), input.dim(0));
	dim3 blockDim = getBlockDim(input.dim(2), input.dim(3), input.dim(0));

	forwardKernel<<<gridDim, blockDim>>>(input, *output_tensor);
	return *output_tensor;
}

Tensor<float, 4> GPUSoftMax::backward(Tensor<float, 4> &error) {
	Tensor<float, 4> next_error({
		error.dim(0), error.dim(1), error.dim(2), error.dim(3) });

	dim3 gridDim = getGridDim(error.dim(2), error.dim(3), error.dim(0));
	dim3 blockDim = getBlockDim(error.dim(2), error.dim(3), error.dim(0));

	backwardKernel<<<gridDim, blockDim>>>(error, *(this->output_tensor), next_error);
	return next_error;
}

