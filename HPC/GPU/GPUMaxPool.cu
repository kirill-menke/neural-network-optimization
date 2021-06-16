#include <cuda.h>
#include <limits>
#include "cuda-utils.h"
#include "GPUMaxPool.h"

__global__
static void forward(Tensor<float, 4> input, Tensor<uint8_t, 5> maximas, Tensor<float, 4> output, int stride, int pool_size) {
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
static void backward(Tensor<float, 4> error_tensor, Tensor<uint8_t, 5> maxiams, Tensor<float, 4> next_error, int stride, int pool_size) {
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
	// TODO
	return input_tensor;
}

Tensor<float, 4> GPUMaxPool::backward(Tensor<float, 4> &error_tensor) {
	// TODO
	return error_tensor;
}




