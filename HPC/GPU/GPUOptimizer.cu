#include <cassert>
#include "./GPUOptimizer.h"
#include "./cuda-utils.h"

__global__
static void update_kernel(Tensor<float, 4> weights, Tensor<float, 1> bias, float learning_rate,
		Tensor<float, 4> gradient_weights, Tensor<float, 1> gradient_bias) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int f = threadIdx.z + blockIdx.z * blockDim.z;

	if (f >= weights.dim(0) || x >= weights.dim(2) || y >= weights.dim(3))
		return;

	int input_channels = weights.dim(1);
	for (int c = 0; c < input_channels; c++)
		weights(f, c, x, y) -= learning_rate * gradient_weights(f, c, x, y);

	if (x == 0 && y == 0)
		bias(f) -= learning_rate * gradient_bias(f);
}

void GPUSgd::update(Tensor<float, 4> weights, Tensor<float, 1> bias,
		Tensor<float, 4> gradient_weights, Tensor<float, 1> gradient_bias) {

	assert(weights.dim(0) == bias.dim(0));

	dim3 gridDim = getGridDim(weights.dim(2), weights.dim(3), weights.dim(0));
	dim3 blockDim = getBlockDim(weights.dim(2), weights.dim(3), weights.dim(0));
	update_kernel<<<gridDim, blockDim>>>(weights, bias, learning_rate,
			gradient_weights, gradient_bias);
}

