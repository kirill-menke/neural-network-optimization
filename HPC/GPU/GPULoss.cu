#include <math.h>
#include <assert.h>

#include "GPULoss.h"
#include "cuda-utils.h"



__global__
static void calculate_loss(Tensor<float, 4> pred, Tensor<float, 3> target, Tensor<float, 2> loss) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	const float EPSILON = 1e-9;
	int batch_size = pred.dim(0);
	int output_width = pred.dim(2), 
		output_height = pred.dim(3);

	if (x >= output_width || y >= output_height)
		return;

	float l = 0.;
	for (int b = 0; b < batch_size; b++) {
		int c = target(b, x, y);

		assert(c == 0 || c == 1);

		l += -log(pred(b, c, x, y) + EPSILON);
	}

	loss(x, y) = l;
}

__global__
static void calculate_gradient_loss(Tensor<float, 4> pred, Tensor<float, 3> target, Tensor<float, 4> gradient_loss) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = pred.dim(0),
		channels = pred.dim(1);
	int output_width = pred.dim(2),
		output_height = pred.dim(3);

	if (x >= output_width || y >= output_height || b >= batch_size)
		return;

	int targetClass = target(b, x, y);
	assert(channels == 2 && 0 <= targetClass && targetClass < 2);
	for (int c = 0; c < channels; c++) {
		float t = targetClass == c ? 1. : 0.;
		gradient_loss(b, c, x, y) = -(t / pred(b, c, x, y));
	}
}

Tensor<float, 2> GPUCrossEntropyLoss::forward(Tensor<float, 4>& pred, Tensor<float, 3>& target) {

	Tensor<float, 2> loss( {imageWidth, imageHeight} );
	dim3 blockDim = getBlockDim(imageWidth, imageHeight, 1);
	dim3 gridDim = getGridDim(imageWidth, imageHeight, 1);

	calculate_loss<<<gridDim, blockDim>>> (pred, target, loss);

	return loss;
}

Tensor<float, 4> GPUCrossEntropyLoss::backward(Tensor<float, 4>& pred, Tensor<float, 3>& target) {

	int batch_size = pred.dim(0),
		channels = pred.dim(1);
	Tensor<float, 4> gradient_loss( { batch_size, channels, imageWidth, imageHeight } );
	
	dim3 blockDim = getBlockDim(imageWidth, imageHeight, batch_size);
	dim3 gridDim = getGridDim(imageWidth, imageHeight, batch_size);
	
	calculate_gradient_loss<<<gridDim, blockDim>>> (pred, target, gradient_loss);

	return gradient_loss;
}