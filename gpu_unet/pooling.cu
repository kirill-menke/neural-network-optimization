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


__global__
void transposed_conv_forward(Tensor<float, 4> input, Tensor<float, 4> output, int strideX, int strideY) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = input.dim(0), channels = input.dim(1),
		inputWidth = input.dim(2), inputHeight = input.dim(3),
		outputWidth = output.dim(2), outputHeight = output.dim(3);

	if (x >= outputWidth || y >= outputHeight || b >= batchSize)
		return;

	for (int c = 0; c < channels; c++) {
		if (x % strideX == 0 && y % strideY == 0 && x / strideX < inputWidth && y / strideY < inputHeight)
			output(b, c, x, y) = input(b, c, x / strideX, y / strideY);
		else
			output(b, c, x, y) = 0.0;
	}
}


__global__
void transposed_conv_backward(Tensor<float, 4> error_tensor, Tensor<float, 4> upsampled_error_tensor,
	Tensor<float, 4> input_tensor, Tensor<float, 4> gradient_weights, Tensor<float, 1> gradient_bias, int strideX, int strideY) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int f = threadIdx.z + blockIdx.z * blockDim.z;

	int filterWidth = gradient_weights.dim(2), filterHeight = gradient_weights.dim(3), numFilters = gradient_weights.dim(0);
	if (i >= filterWidth || j >= filterHeight || f >= numFilters)
		return;

	int batchSize = input_tensor.dim(0), inputChannels = input_tensor.dim(1),
		outputWidth = error_tensor.dim(2), outputHeight = error_tensor.dim(3);

	int w = upsampled_error_tensor.dim(2),
		h = upsampled_error_tensor.dim(3);

	// Calc weights:
	for (int c = 0; c < inputChannels; c++) {
		float err = 0.;

		for (int b = 0; b < batchSize; b++)
			for (int x = 0; x < w; x += strideX)
				for (int y = 0; y < h; y += strideY)
					err += input_tensor(b, c, x + i, y + j) * upsampled_error_tensor(b, f, x, y);

		gradient_weights(f, c, i, j) = err;
	}

	// Calc bias:
	if (i != 0 || j != 0)
		return;

	float err = 0.;
	for (int b = 0; b < batchSize; b++)
		for (int x = 0; x < outputWidth; x++)
			for (int y = 0; y < outputHeight; y++)
				err += error_tensor(b, f, x, y);

	gradient_bias(f) = err;
}


__global__
static void addPadding(Tensor<float, 4> input, Tensor<float, 4> output, int px, int py) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = input.dim(0), channels = input.dim(1),
		inputWidth = input.dim(2), inputHeight = input.dim(3),
		outputWidth = output.dim(2), outputHeight = output.dim(3);

	if (x >= outputWidth || y >= outputHeight)
		return;

	for (int c = 0; c < channels; c++) {
		if (x < px || y < py || x >= px + inputWidth || y >= py + inputHeight)
			output(b, c, x, y) = 0.0;
		else
			output(b, c, x, y) = input(b, c, x - px, y - py);
	}
}


template<bool backward>
__global__
static void convolution(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias, int strideX, int strideY) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = input.dim(0), inputChannels = input.dim(1),
		outputWidth = output.dim(2), outputHeight = output.dim(3),
		outputChannels = output.dim(1);

	int filterWidth = weights.dim(2), filterHeight = weights.dim(3);

	if (b > batchSize || x >= outputWidth || y >= outputHeight)
		return;

	for (int cout = 0; cout < outputChannels; cout++) {
		float value = 0;

		for (int cin = 0; cin < inputChannels; cin++) {
			for (int i = 0; i < filterWidth; i++) {
				for (int j = 0; j < filterHeight; j++) {
					float inputVal = input(b, cin, x * strideX + i, y * strideY + j);
					if (backward)
						// value += inputVal * weights.flipped(cin, cout, i, j);
						printf("Error: flipped not implemented");
					else
						value += inputVal * weights(cout, cin, i, j);
				}
			}
		}

		if (!backward)
			value += bias(cout);

		output(b, cout, x, y) = value;
	}
}


__global__
static void backwardGradients(Tensor<float, 4> error_tensor, Tensor<float, 4> upsampled_error_tensor,
	Tensor<float, 4> input_tensor,
	Tensor<float, 4> gradient_weights, Tensor<float, 1> gradient_bias, int strideX, int strideY) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int f = threadIdx.z + blockIdx.z * blockDim.z;

	int filterWidth = gradient_weights.dim(2), filterHeight = gradient_weights.dim(3), numFilters = gradient_weights.dim(0);
	if (i >= filterWidth || j >= filterHeight || f >= numFilters)
		return;

	int batchSize = input_tensor.dim(0), inputChannels = input_tensor.dim(1),
		outputWidth = error_tensor.dim(2), outputHeight = error_tensor.dim(3);

	int w = upsampled_error_tensor.dim(2),
		h = upsampled_error_tensor.dim(3);

	// Calc weights:
	for (int c = 0; c < inputChannels; c++) {
		float err = 0.;

		for (int b = 0; b < batchSize; b++)
			for (int x = 0; x < w; x += strideX)
				for (int y = 0; y < h; y += strideY)
					err += input_tensor(b, c, x + i, y + j) * upsampled_error_tensor(b, f, x, y);

		gradient_weights(f, c, i, j) = err;
	}

	// Calc bias:
	if (i != 0 || j != 0)
		return;

	float err = 0.;
	for (int b = 0; b < batchSize; b++)
		for (int x = 0; x < outputWidth; x++)
			for (int y = 0; y < outputHeight; y++)
				err += error_tensor(b, f, x, y);

	gradient_bias(f) = err;
}



Tensor<float, 4> TransposedConv::forward(Tensor<float, 4>& input_tensor) {
	int batchSize = input_tensor.dim(0),
		channels = input_tensor.dim(1),
		imageWidth = input_tensor.dim(2),
		imageHeight = input_tensor.dim(3);

	int upsampledWidth = imageWidth * strideX - (strideX - 1),
		upsampledHeight = imageHeight * strideY - (strideY - 1),
		paddingWidth = 2 * (filterSize - 1),
		paddingHeight = 2 * (filterSize - 1),
		outputWidth = strideX * (imageWidth - 1) + weights.dim(2),
		outputHeight = strideY * (imageHeight - 1) + weights.dim(3);

	dim3 gridDim;
	dim3 blockDim;

	Tensor<float, 4> upsampled_tensor({ batchSize, outputChannels, upsampledWidth, upsampledHeight });
	{
		getGridSize(gridDim, blockDim, channels, imageWidth, imageHeight, 32);
		transposed_conv_forward<<<gridDim, blockDim>>> (input_tensor, upsampled_tensor, strideX, strideY);
	}


	if (this->padded_input == nullptr) {
		this->padded_input = new Tensor<float, 4>({ batchSize, outputChannels, upsampledWidth + paddingWidth, upsampledHeight + paddingHeight });
	}
	{
		getGridSize(gridDim, blockDim, channels, padded_input->dim(2), padded_input->dim(3), 32);
		addPadding<<<gridDim, blockDim>>> (upsampled_tensor, *padded_input, filterSize - 1, filterSize - 1);
	}


	Tensor<float, 4> output_tensor({ batchSize, inputChannels, outputWidth, outputHeight });
	{
		getGridSize(gridDim, blockDim, channels, upsampledWidth + 1, upsampledHeight + 1, 32);
		convolution<false><<<gridDim, blockDim>>> (*padded_input, output_tensor, weights, bias, 1, 1);
	}

	return output_tensor;

}


Tensor<float, 4> TransposedConv::backward(Tensor<float, 4>& error_tensor) {
	int batchSize = error_tensor.dim(0),
		channels = error_tensor.dim(1),
		imageWidth = error_tensor.dim(2),
		imageHeight = error_tensor.dim(3);

	dim3 gridDim;
	dim3 blockDim;

	Tensor<float, 4> next_error_tensor({ batchSize, outputChannels, imageWidth, imageHeight });
	{
		getGridSize(gridDim, blockDim, outputChannels, imageWidth, imageHeight, 32);
		convolution<true><<<gridDim, blockDim>>> (error_tensor, next_error_tensor, weights, bias, strideX, strideY);
	}


	Tensor<float, 4> gradient_weights({ outputChannels, inputChannels, filterSize, filterSize});
	Tensor<float, 1> gradient_bias({ outputChannels });
	{
		getGridSize(gridDim, blockDim, outputChannels, filterSize, filterSize, 32);
		backwardGradients<<<gridDim, blockDim>>> (error_tensor, error_tensor, *padded_input, gradient_weights, gradient_bias, strideX, strideY);
	}


	if (this->optimizer != nullptr)
		this->optimizer->update(weights, bias, gradient_weights, gradient_bias);


	return next_error_tensor;
}


