#include "./conv-softmax.h"

#include <cassert>

__global__
static void convolution_softmax_forward(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias) {
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int b = blockIdx.z * blockDim.z + threadIdx.z;

	int batch_size = input.dim(0),
	    input_channels = input.dim(1),
	    output_channels = output.dim(1),
	    width = output.dim(2),
	    height = output.dim(3);

	if (x >= width || y >= height || b >= batch_size)
		return;

	float max = -1e38f;
	for (int c_out = 0; c_out < output_channels; c_out++) {
		float val = bias(c_out);

		// 1x1 Convolution:
		for (int c_in = 0; c_in < input_channels; c_in++) {
			val += input(b, c_in, x, y) * weights(c_out, c_in, 0, 0);
		}

		output(b, c_out, x, y) = val;
		max = fmaxf(max, val);
	}

	// SoftMax:
	float sum = 0.;
	for (int c = 0; c < output_channels; c++) {
		float val = expf(output(b, c, x, y) - max);
		output(b, c, x, y) = val;
		sum += val;
	}

	for (int c = 0; c < output_channels; c++)
		output(b, c, x, y) /= sum;
}

__global__
static void softmax_convolution_backward(Tensor<float, 4> error, Tensor<float, 4> output,
		Tensor<float, 4> next_error, Tensor<float, 4> weights) {
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int b = blockIdx.z * blockDim.z + threadIdx.z;

	int batch_size = error.dim(0),
	    input_channels = next_error.dim(1),
	    output_channels = error.dim(1),
	    width = next_error.dim(2),
	    height = next_error.dim(3);

	if (x >= width || y >= height || b >= batch_size)
		return;

	// SoftMax:
	float sum = 0.;
	for (int c = 0; c < output_channels; c++)
		sum += error(b, c, x, y) * output(b, c, x, y);

	for (int c = 0; c < output_channels; c++)
		error(b, c, x, y) = output(b, c, x, y) * (error(b, c, x, y) - sum);

	// 1x1 Convolution:
	for (int c_in = 0; c_in < input_channels; c_in++) {
		float val = 0.;

		for (int c_out = 0; c_out < output_channels; c_out++) {
			val += error(b, c_out, x, y) * weights(c_out, c_in, 0, 0);
		}

		next_error(b, c_in, x, y) = val;
	}
}

__global__
static void convolution_gradient_weights_reduction(
		Tensor<float, 4> input, Tensor<float, 4> error,
		Tensor<float, 4> gradient_weights) {
	const int x = blockIdx.y * blockDim.y * 2 + threadIdx.y;
	const int y = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	const int batch_size = input.dim(0),
	    input_channels = input.dim(1),
	    output_channels = error.dim(1),
	    width = error.dim(2),
	    height = error.dim(3);

	const int c_out = z / input_channels, c_in = z % input_channels;
	const int threads_in_block = blockDim.x * blockDim.y;
	const int tid = threadIdx.x + threadIdx.y * blockDim.x;

	assert(c_in < input_channels && c_out < output_channels && x < width && y < height);

	// In den Unets immer true. Artifakt einer der Optimierungen aus dem PDF oben.
	bool val2inbound = (x + blockDim.y < width && y + blockDim.x < height);

	extern __shared__ float sm[];

	float val = 0.;
	for (int b = 0; b < batch_size; b++) {
		val += input(b, c_in, x, y) * error(b, c_out, x, y);
		if (val2inbound)
			val += input(b, c_in, x + blockDim.y, y + blockDim.x) * error(b, c_out, x + blockDim.y, y + blockDim.x);
	}
	sm[tid] = val;

	__syncthreads();

	int n = threads_in_block;
	while (n > 1) {
		n /= 2;
		if (tid < n)
			sm[tid] += sm[tid + n];

		__syncthreads();
	}

	if (tid == 0)
		atomicAdd(&gradient_weights(c_out, c_in, 0, 0), sm[tid]);
}

__global__
static void convolution_gradient_bias_reduction(Tensor<float, 4> error, Tensor<float, 1> gradient_bias) {
	const int x = blockIdx.y * blockDim.y * 2 + threadIdx.y;
	const int y = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	const int c_out = blockIdx.z;

	const int batch_size = error.dim(0),
	    output_channels = error.dim(1),
	    width = error.dim(2),
	    height = error.dim(3);

	assert(blockDim.x == blockDim.y);

	const int tid = threadIdx.x + threadIdx.y * blockDim.x;
	const int threads_in_block = blockDim.x * blockDim.y;
	const bool val1inbound = x < width && y < height;
	const bool val2inbound = x + blockDim.y < width && y + blockDim.x < height;

	extern __shared__ float sm[];

	float val = 0.;
	for (int b = 0; b < batch_size; b++) {
		if (val1inbound) val += error(b, c_out, x, y);
		if (val2inbound) val += error(b, c_out, x + blockDim.y, y + blockDim.x);
	}
	sm[tid] = val;

	__syncthreads();

	int n = threads_in_block;
	while (n > 1) {
		n /= 2;
		if (tid < n)
			sm[tid] += sm[tid + n];

		__syncthreads();
	}

	if (tid == 0)
		atomicAdd(&gradient_bias(c_out), sm[tid]);
}

Tensor<float, 4> ConvSoftMax::forward(const Tensor<float, 4> &input) {
	dim3 gridDim;
	dim3 blockDim;
	int batch_size = input.dim(0),
	    width = input.dim(2),
	    height = input.dim(3);

	assert(input.dim(1) == input_channels);

	this->input = new Tensor<float, 4>(input);
	this->output = new Tensor<float, 4>(batch_size, output_channels, width, height);
	getGridSize(gridDim, blockDim, batch_size, width, height);

	convolution_softmax_forward<<<gridDim, blockDim>>>(*this->input, *this->output, weights, bias);
	return *this->output;
}

Tensor<float, 4> ConvSoftMax::backward(const Tensor<float, 4> &error) {
	dim3 gridDim;
	dim3 blockDim;
	int batch_size = error.dim(0),
	    width = error.dim(2),
	    height = error.dim(3);

	assert(error.dim(1) == output_channels);
	assert(input->dim(2) == width);
	assert(input->dim(3) == height);

	Tensor<float, 4> next_error(batch_size, input_channels, width, height);
	getGridSize(gridDim, blockDim, batch_size, width, height);
	softmax_convolution_backward<<<gridDim, blockDim>>>(error, *output, next_error, weights);

	// ACHTUNG: Was Optimierungen angeht, so kann man hier NICHT convolution_backward
	// und die folgenden zwei Kernels gleichzeitig machen, weil in convolution_backward
	// error so verändert wird das der Fehler vom SoftMax drinsteht.

	Tensor<float, 4> gradient_weights(output_channels, input_channels, 1, 1);
	Tensor<float, 1> gradient_bias(output_channels);
	gradient_weights.setZero();
	gradient_bias.setZero();

	// Komischer Shit der mit der parellelen Reduktion zutun hat.
	blockDim.x = width < 32 ? width : 32;
	blockDim.y = height < 32 ? height : 32;
	blockDim.z = 1;
	gridDim.x = ((height + blockDim.x - 1) / blockDim.x) / 2;
	gridDim.y = ((width  + blockDim.y - 1) / blockDim.y) / 2;
	gridDim.z = output_channels * input_channels;
	// Sonderfall bei 16x16 oder kleiner Bildern.
	if (gridDim.x == 0) gridDim.x = 1;
	if (gridDim.y == 0) gridDim.y = 1;

	size_t sm = blockDim.x * blockDim.y * sizeof(float);
	convolution_gradient_weights_reduction<<<gridDim, blockDim, sm>>>(*input, error, gradient_weights);

	gridDim.z = output_channels;
	convolution_gradient_bias_reduction<<<gridDim, blockDim, sm>>>(error, gradient_bias);

	optimizer->update(weights, bias, gradient_weights, gradient_bias);
	delete this->input;
	delete this->output;
	this->input = nullptr;
	this->output = nullptr;
	return next_error;
}


