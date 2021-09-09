#include <cassert>

#include "./optimizer.h"

__global__
static void update_kernel_sgd(Tensor<float, 4> weights, Tensor<float, 1> bias, float learning_rate, 
	Tensor<float, 4> gradient_weights, Tensor<float, 1> gradient_bias) {

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


__global__
void update_kernel_sgd_with_momentum(Tensor<float, 4> weights, Tensor<float, 1> bias, Tensor<float, 4> gradient_weights, Tensor<float, 1> gradient_bias,
	Tensor<float, 4> v_weight, Tensor<float, 1> v_bias, float learning_rate, float momentum_rate) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int f = threadIdx.z + blockIdx.z * blockDim.z;

	if (f >= weights.dim(0) || x >= weights.dim(2) || y >= weights.dim(3))
		return;

	int input_channels = weights.dim(1);
	for (int c = 0; c < input_channels; c++) {
		v_weight(f, c, x, y) = momentum_rate * v_weight(f, c, x, y) - learning_rate * gradient_weights(f, c, x, y);
		weights(f, c, x, y) += v_weight(f, c, x, y);
	}


	if (x == 0 && y == 0) {
		v_bias(f) = momentum_rate * v_bias(f) - learning_rate * gradient_bias(f);
		bias(f) += v_bias(f);
	}
}


__global__
static void update_kernel_adam(Tensor<float, 4> weights, Tensor<float, 1> bias, float learning_rate, float mu, float rho, int iter,
	Tensor<float, 4> gradient_weights, Tensor<float, 1> gradient_bias, Tensor<float, 4> v_weight, Tensor<float, 4> r_weight,
	Tensor<float, 1> v_bias, Tensor<float, 1> r_bias) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int f = threadIdx.z + blockIdx.z * blockDim.z;

	if (f >= weights.dim(0) || x >= weights.dim(2) || y >= weights.dim(3))
		return;

	int input_channels = weights.dim(1);
	for (int c = 0; c < input_channels; c++) {
		v_weight(f, c, x, y) = mu * v_weight(f, c, x, y) + (1 - mu) * gradient_weights(f, c, x, y);
		r_weight(f, c, x, y) = rho * r_weight(f, c, x, y) + (1 - rho) * powf(gradient_weights(f, c, x, y), 2);

		float v_corr = v_weight(f, c, x, y) * (1 / (1 - powf(mu, iter)));
		float r_corr = r_weight(f, c, x, y) * (1 / (1 - powf(rho, iter)));

		weights(f, c, x, y) -= learning_rate * v_corr / (sqrtf(r_corr) + 1e-8);
	}


	if (x == 0 && y == 0) {
		v_bias(f) = mu * v_bias(f) + (1 - mu) * gradient_bias(f);
		r_bias(f) = rho * r_bias(f) + (1 - rho) * powf(gradient_bias(f), 2);

		float v_corr = v_bias(f) * (1 / (1 - powf(mu, iter)));
		float r_corr = r_bias(f) * (1 / (1 - powf(rho, iter)));

		bias(f) -= learning_rate * v_corr / (sqrtf(r_corr) + 1e-8);
	}

}


void Sgd::update(Tensor<float, 4> &weights, Tensor<float, 1> &bias, Tensor<float, 4> &gradient_weights, Tensor<float, 1> &gradient_bias) {
	assert(weights.dim(0) == bias.dim(0));

	// gradient_weights.moveToHost().dump4D(stdout, "gradient_weights");

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, weights.dim(0), weights.dim(2), weights.dim(3));
	update_kernel_sgd<<<gridDim, blockDim>>>(weights, bias, learning_rate, gradient_weights, gradient_bias);
}


void SgdWithMomentum::update(Tensor<float, 4> &weights, Tensor<float, 1> &bias,
	Tensor<float, 4> &gradient_weights, Tensor<float, 1> &gradient_bias) {

	assert(weights.dim(0) == bias.dim(0));

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, weights.dim(0), weights.dim(2), weights.dim(3));

	update_kernel_sgd_with_momentum<<<gridDim, blockDim>>> (weights, bias, gradient_weights, gradient_bias,
		v_weight, v_bias, learning_rate, momentum_rate);
}


void Adam::update(Tensor<float, 4> &weights, Tensor<float, 1> &bias,
	Tensor<float, 4> &gradient_weights, Tensor<float, 1> &gradient_bias) {

	assert(weights.dim(0) == bias.dim(0));

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, weights.dim(0), weights.dim(2), weights.dim(3));

	update_kernel_adam<<<gridDim, blockDim>>> (weights, bias, learning_rate, mu, rho, iter,
		gradient_weights, gradient_bias, v_weight, r_weight, v_bias, r_bias);

	iter++;
}