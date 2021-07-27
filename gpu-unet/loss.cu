#include "./loss.h"

#include <cassert>
#include <cmath>

#define EPSILON 1e-07f

__global__
static void cel_forward(Tensor<float, 4> prediction, Tensor<float, 4> truth, Tensor<float, 2> loss) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int batch_size = prediction.dim(0),
	    channels = prediction.dim(1),
	    width = prediction.dim(2),
	    height = prediction.dim(3);

	if (x >= width || y >= height)
		return;

	float l = 0.;

	for (int b = 0; b < batch_size; b++) {
		int classification = -1;
		for (int c = 0; c < channels; c++) {
			if (truth(b, c, x, y) == 1.) {
				classification = c;
			}
		}

		assert(classification != -1);

		l += -logf(prediction(b, classification, x, y) + EPSILON);
	}

	loss(x, y) = l;
}

__global__
static void cel_backward(Tensor<float, 4> prediction, Tensor<float, 4> truth, Tensor<float, 4> error) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int batch_size = prediction.dim(0),
	    channels = prediction.dim(1),
	    width = prediction.dim(2),
	    height = prediction.dim(3);

	if (x >= width || y >= height)
		return;

	for (int b = 0; b < batch_size; b++) {
		for (int c = 0; c < channels; c++) {
			error(b, c, x, y) = -(truth(b, c, x, y) / prediction(b, c, x, y));
		}
	}
}

Tensor<float, 2> PerPixelCELoss::forward(const Tensor<float, 4> &prediction, const Tensor<float, 4> &truth) {
	dim3 gridDim;
	dim3 blockDim;
	int batch_size = prediction.dim(0),
	    channels = prediction.dim(1),
	    width = prediction.dim(2),
	    height = prediction.dim(3);

	assert(batch_size == truth.dim(0)
		&& channels == truth.dim(1)
		&& width == truth.dim(2)
		&& height == truth.dim(3));

	Tensor<float, 2> loss(width, height);
	getGridSize(gridDim, blockDim, 1, width, height);
	cel_forward<<<gridDim, blockDim>>>(prediction, truth, loss);

	return loss;
}

Tensor<float, 4> PerPixelCELoss::backward(const Tensor<float, 4> &prediction, const Tensor<float, 4> &truth) {
	dim3 gridDim;
	dim3 blockDim;
	int batch_size = prediction.dim(0),
	    channels = prediction.dim(1),
	    width = prediction.dim(2),
	    height = prediction.dim(3);

	Tensor<float, 4> error(batch_size, channels, width, height);
	getGridSize(gridDim, blockDim, 1, width, height);
	cel_backward<<<gridDim, blockDim>>>(prediction, truth, error);

	return error;
}


