#include "./unet-up.h"

template<int pool_size>
__global__
static void unet_up_forward(Tensor<float, 4> input1, Tensor<float, 4> input2, Tensor<float, 4> output) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = input1.dim(0),
	    channels1 = input1.dim(1),
	    channels2 = input2.dim(1),
	    width = input2.dim(2),
	    height = input2.dim(3);

	int x_out = x * pool_size,
	    y_out = y * pool_size;

	if (x >= width || y >= height)
		return;

	if (c < channels1) {
		for (int b = 0; b < batch_size; b++)
			for (int i = 0; i < pool_size; i++)
				for (int j = 0; j < pool_size; j++)
					output(b, c, x_out + i, y_out + j) = input1(b, c, x_out + i, y_out + j);
	} else if (c < channels1 + channels2) {
		for (int b = 0; b < batch_size; b++) {
			int val = input2(b, c - channels1, x, y);

			for (int i = 0; i < pool_size; i++)
				for (int j = 0; j < pool_size; j++)
					output(b, c, x_out + i, y_out + j) = val;
		}
	}
}

template<int pool_size>
__global__
static void unet_up_backward(Tensor<float, 4> error, Tensor<float, 4> next_error, int skipcon_channels, int upsample_channels) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = error.dim(0),
	    input_width = next_error.dim(2),
	    input_height = next_error.dim(3);

	if (x >= input_width || y >= input_height || c >= upsample_channels)
		return;

	int x_out = x * pool_size,
	    y_out = y * pool_size;

	for (int b = 0; b < batch_size; b++) {
		float val = 0.;
		for (int i = 0; i < pool_size; i++) {
			for (int j = 0; j < pool_size; j++) {
				val += error(b, skipcon_channels + c, x_out + i, y_out + j);
			}
		}

		next_error(b, c, x, y) = val / (pool_size * pool_size);
	}
}

Tensor<float, 4> UnetUp::forward(const Tensor<float, 4> &input1, const Tensor<float, 4> &input2) {
	int batch_size = input1.dim(0),
	    channels1 = input1.dim(1),
		channels2 = input2.dim(1),
	    output_width = input1.dim(2),
	    output_height = input1.dim(3);

	int input_width = output_width / pool_size,
	    input_height = output_height / pool_size;

	assert(input2.dim(0) == batch_size
		&& input2.dim(2) == input_width
		&& input2.dim(3) == input_height);

	assert(channels1 == skipcon_channels
		&& channels2 == upsample_channels);

	Tensor<float, 4> output_tensor(batch_size, channels1 + channels2, output_width, output_height);

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, channels1 + channels2, input_width, input_height, 32);

	switch (pool_size) {
	case 2:
		unet_up_forward<2><<<gridDim, blockDim>>>(input1, input2, output_tensor);
		break;
	default: assert(false);
	}

	return output_tensor;
}

Tensor<float, 4> UnetUp::backward(const Tensor<float, 4> &error) {
	int batch_size = error.dim(0),
		channels = error.dim(1),
		output_width = error.dim(2),
		output_height = error.dim(3);

	int input_width = output_width / pool_size,
		input_height = output_height / pool_size;

	assert(channels == skipcon_channels + upsample_channels);

	Tensor<float, 4> next_error(batch_size, upsample_channels, input_width, input_height);

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, upsample_channels, input_width, input_height, 32);

	switch (pool_size) {
	case 2:
		unet_up_backward<2><<<gridDim, blockDim>>>(error, next_error, skipcon_channels, upsample_channels);
		break;
	default: assert(false);
	}


	return next_error;
}




