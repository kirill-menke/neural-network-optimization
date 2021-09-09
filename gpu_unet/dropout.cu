#include <curand.h>
#include <curand_kernel.h>

#include "dropout.h"


__global__
void dropout_forward(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> mask, float prob) {
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = input.dim(0),
		channels = input.dim(1),
		width = input.dim(2),
		height = input.dim(3);

	int tid = c + channels * (x + width * y);

	if (c >= channels || x >= width || y >= height)
		return;

	curandState_t state;
	curand_init(0, tid, 0, &state);

	for (int b = 0; b < batch_size; b++) {
		float rnd = curand_uniform(&state);
		int activation = rnd > prob;
		mask(b, c, x, y) = activation;
		output(b, c, x, y) = input(b, c, x, y) * activation / prob;
	}
}


__global__
void dropout_backward(Tensor<float, 4> error_tensor, Tensor<float, 4> next_error_tensor, Tensor<float, 4> mask) {
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = error_tensor.dim(0),
		channels = error_tensor.dim(1),
		width = error_tensor.dim(2),
		height = error_tensor.dim(3);

	if (c >= channels || x >= width || y >= height)
		return;

	for (int b = 0; b < batch_size; b++) {
		next_error_tensor(b, c, x, y) = error_tensor(b, c, x, y) * mask(b, c, x, y);
	}
}


Tensor<float, 4> Dropout::forward(Tensor<float, 4> const & input) {
	if (testing_phase)
		return input;

	int batch_size = input.dim(0),
		channels = input.dim(1),
		width = input.dim(2),
		height = input.dim(3);

	if (!this->mask)
		this->mask = std::make_shared<Tensor<float, 4>>(batch_size, channels, width, height);
	Tensor<float, 4> output(batch_size, channels, width, height);

	dim3 gridDim, blockDim;
	getGridSize(gridDim, blockDim, channels, width, height);
	dropout_forward<<<gridDim, blockDim>>> (input, *mask, output, 1 - probability);

	return input;
}


Tensor<float, 4> Dropout::backward(Tensor<float, 4> const & error_tensor) {
	int batch_size = error_tensor.dim(0),
		channels = error_tensor.dim(1),
		width = error_tensor.dim(2),
		height = error_tensor.dim(3);
	
	Tensor<float, 4> next_error_tensor(batch_size, channels, width, height);

	dim3 gridDim, blockDim;
	getGridSize(gridDim, blockDim, channels, width, height);
	dropout_backward<<<gridDim, blockDim>>> (error_tensor, next_error_tensor, *mask);

	return next_error_tensor;
}