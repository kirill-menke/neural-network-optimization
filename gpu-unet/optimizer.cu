#include <cassert>

#include "./optimizer.h"

__global__
static void update_kernel(Tensor<float, 4> weights, Tensor<float, 1> bias, float learning_rate, Tensor<float, 4> gradient_weights, Tensor<float, 1> gradient_bias) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int f = threadIdx.z + blockIdx.z * blockDim.z;

	if (f >= weights.dim(0) || x >= weights.dim(2) || y >= weights.dim(3))
		return;

	int channels = weights.dim(1);
	for (int c = 0; c < channels; c++)
		weights(f, c, x, y) -= learning_rate * gradient_weights(f, c, x, y);

	if (x == 0 && y == 0)
		bias(f) -= learning_rate * gradient_bias(f);
}

void Sgd::update(Tensor<float, 4> &weights, Tensor<float, 1> &bias, Tensor<float, 4> &gradient_weights, Tensor<float, 1> &gradient_bias) {
	assert(weights.dim(0) == bias.dim(0));

	// gradient_weights.moveToHost().dump4D(stdout, "gradient_weights");

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, weights.dim(0), weights.dim(2), weights.dim(3));
	update_kernel<<<gridDim, blockDim>>>(weights, bias, learning_rate, gradient_weights, gradient_bias);
}

