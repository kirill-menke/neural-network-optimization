#include <cuda.h>
#include <limits>
#include <assert.h>

#include "cuda-utils.h"
#include "GPUMaxPool.h"

__global__
static void forwardKernel(Tensor<float, 4> input, Tensor<uint8_t, 5> maximas, Tensor<float, 4> output, int stride, int pool_size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = input.dim(0), channels = input.dim(1),
	    input_width = input.dim(2), input_height = input.dim(3);
	int output_width = output.dim(2), output_height = output.dim(3);
	if (x >= output_width || y >= output_height || b >= batch_size)
		return;

	for (int c = 0; c < channels; c++) {
		int x_in = x * stride, y_in = y * stride;
		float max = -std::numeric_limits<float>::infinity();
		int i_max = 0, j_max = 0;

		for (int i = 0; i < pool_size; i++) {
			for (int j = 0; j < pool_size; j++) {
				float val = input(b, c, x_in + i, y_in + j);
				if (val > max) {
					max = val;
					i_max = i;
					j_max = j;
				}
			}
		}

		maximas(b, c, x, y, 0) = i_max;
		maximas(b, c, x, y, 1) = j_max;
		output(b, c, x, y) = max;
	}
}

/*
 * When called, next_error MUST be initialized with zeros!
 */
__global__
static void backwardKernel(Tensor<float, 4> error_tensor, Tensor<uint8_t, 5> maxiams, Tensor<float, 4> next_error, int stride, int pool_size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = error_tensor.dim(0), channels = error_tensor.dim(1),
	    input_width = next_error.dim(2), input_height = next_error.dim(3);
	int output_width = error_tensor.dim(2), output_height = error_tensor.dim(3);
	if (x >= output_width || y >= output_height || b >= batch_size)
		return;

	for (int c = 0; c < channels; c++) {
		float err = error_tensor(b, c, x, y);
		int i = maxiams(b, c, x, y, 0);
		int j = maxiams(b, c, x, y, 1);

		// If we would allow overlapping kernels, we would need atomicAdd!
		next_error(b, c, x * stride + i, y * stride + j) = err;
	}

}

Tensor<float, 4> GPUMaxPool::forward(Tensor<float, 4> &input_tensor) {
	assert(input_tensor.dim(0) == batch_size && input_tensor.dim(1) == channels
		&& input_tensor.dim(2) == input_width && input_tensor.dim(3) == input_height);
	int output_width = this->outputWidth(), output_height = this->outputHeight();

	Tensor<float, 4> output_tensor(Tensor<float, 4>::ON_GPU, {
			batch_size, channels, output_width, output_height });

	dim3 gridDim = getGridDim(output_width, output_height, batch_size);
	dim3 blockDim = getBlockDim(output_width, output_height, batch_size);

	forwardKernel<<<gridDim, blockDim>>>(input_tensor, maximas, output_tensor, stride, pool_size);

	return output_tensor;
}

Tensor<float, 4> GPUMaxPool::backward(Tensor<float, 4> &error_tensor) {
	int output_width = this->outputWidth(), output_height = this->outputHeight();
	assert(error_tensor.dim(0) == batch_size && error_tensor.dim(1) == channels
		&& error_tensor.dim(2) == output_width && error_tensor.dim(3) == output_height);

	Tensor<float, 4> next_error(Tensor<float, 4>::ON_GPU, {
			batch_size, channels, input_width, input_height });
	next_error.setZero(true);

	dim3 gridDim = getGridDim(output_width, output_height, batch_size);
	dim3 blockDim = getBlockDim(output_width, output_height, batch_size);

	backwardKernel<<<gridDim, blockDim>>>(error_tensor, maximas, next_error, stride, pool_size);

	return next_error;
}




