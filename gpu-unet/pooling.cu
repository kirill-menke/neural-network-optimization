#include "./pooling.h"

__global__
static void maxpool_forward(Tensor<float, 4> input, Tensor<uint8_t, 5> maximas, Tensor<float, 4> output, int pool_size) {
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
static void maxpool_backward(Tensor<float, 4> error_tensor, Tensor<uint8_t, 5> maximas, Tensor<float, 4> next_error, int pool_size) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = error_tensor.dim(0),
	    channels = error_tensor.dim(1),
	    output_width = error_tensor.dim(2),
	    output_height = error_tensor.dim(3);

	if (x >= output_width || y >= output_height || c >= channels)
		return;

	for (int b = 0; b < batch_size; b++) {
		for (int i = 0; i < pool_size; i++) {
			for (int j = 0; j < pool_size; j++) {
				next_error(b, c, x * pool_size + i, y * pool_size + j) = 0.;
			}
		}

		int i = maximas(b, c, x, y, 0);
		int j = maximas(b, c, x, y, 1);
		next_error(b, c, x * pool_size + i, y * pool_size + j) = error_tensor(b, c, x, y);
	}
}

Tensor<float, 4> MaxPool::forward(const Tensor<float, 4> &input_tensor) {
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
	maxpool_forward<<<gridDim, blockDim>>>(input_tensor, *maximas, output_tensor, pool_size);

	return output_tensor;
}

Tensor<float, 4> MaxPool::backward(const Tensor<float, 4> &error_tensor) {
	int batch_size = error_tensor.dim(0),
	    channels = error_tensor.dim(1),
	    output_width = error_tensor.dim(2),
	    output_height = error_tensor.dim(3);

	int input_width = output_width * pool_size,
	    input_height = output_height * pool_size;

	assert(maximas->dim(0) == batch_size
		&& maximas->dim(1) == channels
		&& maximas->dim(2) == output_width
		&& maximas->dim(3) == output_height
		&& maximas->dim(4) == 2);

	Tensor<float, 4> next_error(batch_size, channels, input_width, input_height);
	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, channels, output_width, output_height, 32);
	maxpool_backward<<<gridDim, blockDim>>>(error_tensor, *maximas, next_error, pool_size);

	delete maximas;
	return next_error;
}

__global__
static void upsample_forward(Tensor<float, 4> input, Tensor<float, 4> output, int pool_size) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = input.dim(0),
	    channels = input.dim(1),
	    input_width = input.dim(2),
	    input_height = input.dim(3);

	if (x >= input_width || y >= input_height || c >= channels)
		return;

	int x_out = x * pool_size,
	    y_out = y * pool_size;

	for (int b = 0; b < batch_size; b++) {
		float val = input(b, c, x, y);
		for (int i = 0; i < pool_size; i++) {
			for (int j = 0; j < pool_size; j++) {
				output(b, c, x_out + i, y_out + j) = val;
			}
		}
	}
}

__global__
static void upsample_backward(Tensor<float, 4> error, Tensor<float, 4> next_error, int pool_size) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = error.dim(0),
	    channels = error.dim(1),
	    input_width = next_error.dim(2),
	    input_height = next_error.dim(3);

	if (x >= input_width || y >= input_height || c >= channels)
		return;

	int x_out = x * pool_size,
	    y_out = y * pool_size;

	for (int b = 0; b < batch_size; b++) {
		float val = 0.;
		for (int i = 0; i < pool_size; i++) {
			for (int j = 0; j < pool_size; j++) {
				val += error(b, c, x_out + i, y_out + j);
			}
		}

		next_error(b, c, x, y) = val / (pool_size * pool_size);
	}
}

Tensor<float, 4> Upsample::forward(const Tensor<float, 4> &input_tensor) {
	int batch_size = input_tensor.dim(0),
	    channels = input_tensor.dim(1),
	    input_width = input_tensor.dim(2),
	    input_height = input_tensor.dim(3);

	int output_width = input_width * pool_size,
	    output_height = input_height * pool_size;

	Tensor<float, 4> output_tensor(batch_size, channels, output_width, output_height);

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, channels, input_width, input_height, 32);
	upsample_forward<<<gridDim, blockDim>>>(input_tensor, output_tensor, pool_size);

	return output_tensor;
}

Tensor<float, 4> Upsample::backward(const Tensor<float, 4> &error_tensor) {
	int batch_size = error_tensor.dim(0),
	    channels = error_tensor.dim(1),
	    output_width = error_tensor.dim(2),
	    output_height = error_tensor.dim(3);

	int input_width = output_width / pool_size,
	    input_height = output_height / pool_size;

	Tensor<float, 4> next_error(batch_size, channels, input_width, input_height);
	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, channels, input_width, input_height, 32);
	upsample_backward<<<gridDim, blockDim>>>(error_tensor, next_error, pool_size);

	return next_error;
}



