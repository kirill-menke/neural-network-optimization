#include "./unet-down.h"

__global__
static void unet_down_forward(Tensor<float, 4> input, Tensor<uint8_t, 5> maximas, Tensor<float, 4> output, int pool_size) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = input.dim(0), channels = input.dim(1),
	    input_width = input.dim(2), input_height = input.dim(3);
	int output_width = output.dim(2), output_height = output.dim(3);

	if (x >= output_width || y >= output_height || c >= channels)
		return;

	int x_in = x * pool_size,
	    y_in = y * pool_size;

	for (int b = 0; b < batch_size; b++) {
		float max = input(b, c, x_in, y_in);
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

__global__
static void unet_down_backward(Tensor<float, 4> error1, Tensor<float, 4> error2, Tensor<uint8_t, 5> maximas, Tensor<float, 4> next_error, int pool_size) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = error1.dim(0),
	    channels = error1.dim(1),
	    output_width = error1.dim(2),
	    output_height = error1.dim(3);

	if (x >= output_width || y >= output_height || c >= channels)
		return;

	for (int b = 0; b < batch_size; b++) {
		for (int i = 0; i < pool_size; i++) {
			for (int j = 0; j < pool_size; j++) {
				next_error(b, c, x * pool_size + i, y * pool_size + j) = error2(b, c, x * pool_size + i, y * pool_size + j);
			}
		}

		int i = maximas(b, c, x, y, 0);
		int j = maximas(b, c, x, y, 1);
		next_error(b, c, x * pool_size + i, y * pool_size + j) += error1(b, c, x, y);
	}
}

Tensor<float, 4> UnetDown::forward(const Tensor<float, 4> &input_tensor) {
	int batch_size = input_tensor.dim(0),
	    channels = input_tensor.dim(1),
	    input_width = input_tensor.dim(2),
	    input_height = input_tensor.dim(3);

	int output_width = input_width / pool_size,
	    output_height = input_height / pool_size;

	maximas = new Tensor<uint8_t, 5>({ batch_size, channels, output_width, output_height, 2 });

	Tensor<float, 4> output_tensor(batch_size, channels, output_width, output_height);
	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, channels, output_width, output_height, 32);
	unet_down_forward<<<gridDim, blockDim>>>(input_tensor, *maximas, output_tensor, pool_size);

	return output_tensor;
}

Tensor<float, 4> UnetDown::backward(const Tensor<float, 4> &error1, const Tensor<float, 4> &error2) {
	int batch_size = error1.dim(0),
		channels = error1.dim(1),
	    output_width = error1.dim(2),
	    output_height = error1.dim(3);

	int input_width = output_width * pool_size,
	    input_height = output_height * pool_size;

	assert(error2.dim(0) == batch_size
		&& error2.dim(1) > channels
		&& error2.dim(2) == input_width
		&& error2.dim(3) == input_height);

	assert(maximas->dim(0) == batch_size
		&& maximas->dim(1) == channels
		&& maximas->dim(2) == output_width
		&& maximas->dim(3) == output_height
		&& maximas->dim(4) == 2);

	Tensor<float, 4> next_error(batch_size, channels, input_width, input_height);
	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, channels, output_width, output_height, 32);
	unet_down_backward<<<gridDim, blockDim>>>(error1, error2, *maximas, next_error, pool_size);

	delete maximas;
	maximas = nullptr;
	return next_error;
}

