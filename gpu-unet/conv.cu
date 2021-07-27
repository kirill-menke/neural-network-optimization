#include "conv.h"

#include <cassert>

#define NO_EXTRA_PAD 1

/*
 * Für das padding bei convolutions gibts zwei Möglichkeiten:
 * 1.: Einen neuen, leicht größeren Tensor erstellen in eigenem Kernel
 *     der an den Rändern mit 0 befüllt wird.
 * 2.: In den convolution-Kernels beim Zugriff auf den Input/Fehler-Tensor
 *     schauen ob der Zugriff in den Input fällt oder den Rand, falls Rand dann
 *     statt Original-Wert 0 zurückgeben.
 *
 * Zumindest auf den CIP-GPUs ist kein Performance-Unterschied messbar, daher
 * sind hier immernoch beide Optionen.
 */
#if NO_EXTRA_PAD
template<int filter_size>
__device__
static inline float access_padded(Tensor <float, 4> &tensor, int b, int c, int x, int y, int w, int h) {
	x -= filter_size / 2;
	y -= filter_size / 2;
	if (0 <= x && x < w && 0 <= y && y < h)
		return tensor(b, c, x, y);
	return 0.;
}
#else
__global__
static void pad(Tensor<float, 4> input, Tensor<float, 4> output, int padding) {
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.z * blockDim.z + threadIdx.z;

	if (c >= output.dim(1) || x >= output.dim(2) || y >= output.dim(3))
		return;

	int width = input.dim(2), height = input.dim(3);

	int batch_size = input.dim(0);
	for (int b = 0; b < batch_size; b++) {
#if 1
		if (x < padding || y < padding || x >= width + padding || y >= height + padding)
			output(b, c, x, y) = 0.0;
		else
			output(b, c, x, y) = input(b, c, x - padding, y - padding);
#else
		// Statt den Rand auf 0 zu setzen kann man ja eig auch den Nachbar-Pixel nehmen.
		int x_in = x - padding, y_in = y - padding;
		if (x_in < 0) x_in = 0;
		if (y_in < 0) y_in = 0;
		if (x_in >= width)  x_in = width  - 1;
		if (y_in >= height) y_in = height - 1;
		output(b, c, x, y) = input(b, c, x_in, y_in);
#endif
	}
}
#endif

template<int filter_size>
__global__
static void convolution_forward(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias) {
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int c_out = blockIdx.z * blockDim.z + threadIdx.z;

	int batch_size = input.dim(0),
	    input_channels = input.dim(1),
	    output_channels = output.dim(1),
	    width = output.dim(2),
	    height = output.dim(3);

	if (x >= width || y >= height || c_out >= output_channels)
		return;

	float channel_bias = bias(c_out);
	for (int b = 0; b < batch_size; b++) {
		float val = channel_bias;

		for (int c_in = 0; c_in < input_channels; c_in++) {
			for (int i = 0; i < filter_size; i++) {
				for (int j = 0; j < filter_size; j++) {
#if NO_EXTRA_PAD
					float input_val = access_padded<filter_size>(input, b, c_in, x + i, y + j, width, height);
					val += input_val * weights(c_out, c_in, i, j);
#else
					val += input(b, c_in, x + i, y + j) * weights(c_out, c_in, i, j);
#endif
				}
			}
		}

		output(b, c_out, x, y) = val;
	}
}

template<int filter_size>
__global__
static void convolution_backward(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float, 4> weights) {
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int c_in = blockIdx.z * blockDim.z + threadIdx.z;

	int batch_size = error.dim(0),
	    input_channels = next_error.dim(1),
	    output_channels = error.dim(1),
	    width = next_error.dim(2),
	    height = next_error.dim(3);

	if (x >= width || y >= height || c_in >= input_channels)
		return;

	for (int b = 0; b < batch_size; b++) {
		float val = 0.;

		for (int c_out = 0; c_out < output_channels; c_out++) {
			for (int i = 0; i < filter_size; i++) {
				for (int j = 0; j < filter_size; j++) {
#if NO_EXTRA_PAD
					float error_val = access_padded<filter_size>(error, b, c_out, x + i, y + j, width, height);
					val += error_val * weights(c_out, c_in, filter_size - i - 1, filter_size - j - 1);
#else
					val += error(b, c_out, x + i, y + j) * weights(c_out, c_in, filter_size - i - 1, filter_size - j - 1);
#endif
				}
			}
		}

		next_error(b, c_in, x, y) = val;
	}
}

/*
 * Der wohl komplexeste Teil des Conv. Layer.
 * Dieser Kernel rechnet den Gradienten w.r.t. den Gewichten aus.
 * Problem bei trivialer Implementierung: Man startet Kernel für jeden Pixel
 * in den Gewichten und iteriert über das Input-Bild. Das ist MEGA scheiße für die GPU
 * weil VIEL zu wenig Möglichkeiten zur Parallelität.
 *
 * Deswegen machen wir hier was schlaueres: Wir starten Kernel pro Pixel im Input-Bild,
 * und machen dann für jeden Pixel in den Gewichten eine parallele Reduktion der berechneten Werte.
 * Weil das Input-Bild größer als 32x32 Pixel sein kann die hier ein Block abdeckt müssen wir am Ende
 * `atomicAdd`'s machen, und deshalb muss da auch der Speicher vorher genullt sein.
 *
 * Statt den `atomicAdd`s und dem vorherigen nullen könnte man auch noch einen Reduktions-Kernel hinterherstarten
 * der die Zwischen-Reduktionswerte in den `gradient_weights` Tensor schreibt. Das könnte man noch ausprobieren.
 * Andere, teils schon angewendete Optimierungen: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */
template<int filter_size>
__global__
static void convolution_gradient_weights_reduction(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights) {
	const int block_x = blockIdx.y;
	const int block_y = blockIdx.x;
	const int thread_x = threadIdx.y;
	const int thread_y = threadIdx.x;
	const int x = block_x * blockDim.y * 2 + thread_x;
	const int y = block_y * blockDim.x * 2 + thread_y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	const int batch_size = input.dim(0),
	    input_channels = input.dim(1),
	    output_channels = error.dim(1),
	    width = error.dim(2),
	    height = error.dim(3);

	const int c_out = z / input_channels, c_in = z % input_channels;
	const int tid = thread_x * blockDim.y + thread_y;
	const int threads_in_block = blockDim.x * blockDim.y;

	assert(c_in < input_channels && c_out < output_channels && x < width && y < height);

	// In den Unets immer true. Artifakt einer der Optimierungen aus dem PDF oben.
	bool val2inbound = (x + blockDim.y < width && y + blockDim.x < height);

	extern __shared__ float sm[];

	for (int i = 0; i < filter_size; i++) {
		for (int j = 0; j < filter_size; j++) {
			float val = 0.;
			for (int b = 0; b < batch_size; b++) {
#if NO_EXTRA_PAD
				val += access_padded<filter_size>(input, b, c_in, x + i, y + j, width, height) * error(b, c_out, x, y);
				if (val2inbound)
					val += access_padded<filter_size>(input, b, c_in, x + i + blockDim.y, y + j + blockDim.x, width, height) * error(b, c_out, x + blockDim.y, y + blockDim.x);
#else
				val += input(b, c_in, x + i, y + j) * error(b, c_out, x, y);
				if (val2inbound)
					val += input(b, c_in, x + i + blockDim.y, y + j + blockDim.x) * error(b, c_out, x + blockDim.y, y + blockDim.x);
#endif
			}
			sm[tid] = val;

			__syncthreads();

			for (int s = threads_in_block / 2; s > 0; s >>= 1) {
				if (tid < s)
					sm[tid] += sm[tid + s];

				__syncthreads();
			}

			if (tid == 0)
				atomicAdd(&gradient_weights(c_out, c_in, i, j), sm[tid]);
		}
	}
}

/* Siehe oben, quasi das selbe.
 */
__global__
static void convolution_gradient_bias_reduction(Tensor<float, 4> error, Tensor<float, 1> gradient_bias) {
	const int block_x = blockIdx.y;
	const int block_y = blockIdx.x;
	const int thread_x = threadIdx.y;
	const int thread_y = threadIdx.x;
	const int x = block_x * blockDim.y + thread_x;
	const int y = block_y * blockDim.x + thread_y;
	const int c_out = blockIdx.z;

	const int batch_size = error.dim(0),
	    output_channels = error.dim(1),
	    width = error.dim(2),
	    height = error.dim(3);

	const int tid = thread_x * blockDim.y + thread_y;
	const int threads_in_block = blockDim.x * blockDim.y;
	const bool inbound = c_out < output_channels && x < width && y < height;

	extern __shared__ float sm[];

	if (inbound) {
		float val = 0.;
		for (int b = 0; b < batch_size; b++)
			val += error(b, c_out, x, y);
		sm[tid] = val;
	}

	__syncthreads();

	for (int s = 1; s < threads_in_block; s *= 2) {
		int idx = 2 * s * tid;
		if (idx < threads_in_block && inbound) {
			sm[idx] += sm[idx + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(&gradient_bias(c_out), sm[tid]);
	}
}

Tensor<float, 4> Conv::forward(const Tensor<float, 4> &input) {
	dim3 gridDim;
	dim3 blockDim;
	int batch_size = input.dim(0),
	    width = input.dim(2),
	    height = input.dim(3);

	assert(input.dim(1) == input_channels);

#if NO_EXTRA_PAD
	this->input = new Tensor<float, 4>(input);
#else
	int padding = filter_size / 2;
	this->input = new Tensor<float, 4>(batch_size, input_channels, width + 2 * padding, height + 2 * padding);
	getGridSize(gridDim, blockDim, input_channels, width + 2 * padding, height + 2 * padding);
	pad<<<gridDim, blockDim>>>(input, *this->input, padding);
#endif

	Tensor<float, 4> output(batch_size, output_channels, width, height);
	getGridSize(gridDim, blockDim, output_channels, width, height);
	switch (filter_size) {
	case 1:
		convolution_forward<1><<<gridDim, blockDim>>>(*this->input, output, weights, bias);
		break;
	case 3:
		convolution_forward<3><<<gridDim, blockDim>>>(*this->input, output, weights, bias);
		break;
	case 5:
		convolution_forward<5><<<gridDim, blockDim>>>(*this->input, output, weights, bias);
		break;
	default:
		assert(false);
	}
	return output;
}

Tensor<float, 4> Conv::backward(const Tensor<float, 4> &error) {
	dim3 gridDim;
	dim3 blockDim;
	int batch_size = error.dim(0),
	    width = error.dim(2),
	    height = error.dim(3);

#if NO_EXTRA_PAD
	assert(error.dim(1) == output_channels);
	assert(input->dim(2) == width);
	assert(input->dim(3) == height);
	const Tensor<float, 4> &padded_error = error;
#else
	int padding = filter_size / 2;
	assert(error.dim(1) == output_channels);
	assert(input->dim(2) == width + 2 * padding);
	assert(input->dim(3) == height + 2 * padding);
	Tensor<float, 4> padded_error(batch_size, output_channels, width + 2 * padding, height + 2 * padding);
	getGridSize(gridDim, blockDim, output_channels, width + 2 * padding, height + 2 * padding);
	pad<<<gridDim, blockDim>>>(error, padded_error, padding);
#endif

	Tensor<float, 4> next_error(batch_size, input_channels, width, height);
	getGridSize(gridDim, blockDim, input_channels, width, height);
	switch (filter_size) {
	case 1:
		convolution_backward<1><<<gridDim, blockDim>>>(padded_error, next_error, weights);
		break;
	case 3:
		convolution_backward<3><<<gridDim, blockDim>>>(padded_error, next_error, weights);
		break;
	case 5:
		convolution_backward<5><<<gridDim, blockDim>>>(padded_error, next_error, weights);
		break;
	default:
		assert(false);
	}

	Tensor<float, 4> gradient_weights(output_channels, input_channels, filter_size, filter_size);
	Tensor<float, 1> gradient_bias(output_channels);
	gradient_weights.setZero();
	gradient_bias.setZero();

	// Komischer Shit der mit der parellelen Reduktion zutun hat.
	blockDim.x = width < 16 ? width : 16;
	blockDim.y = height < 16 ? height : 16;
	blockDim.z = 1;
	gridDim.x = ((height + blockDim.x - 1) / blockDim.x) / 2;
	gridDim.y = ((width  + blockDim.y - 1) / blockDim.y) / 2;
	gridDim.z = output_channels * input_channels;
	// Sonderfall bei 16x16 oder kleiner Bildern.
	if (gridDim.x == 0) gridDim.x = 1;
	if (gridDim.y == 0) gridDim.y = 1;

	size_t sm = blockDim.x * blockDim.y * sizeof(float);
	switch (filter_size) {
	case 1:
		convolution_gradient_weights_reduction<1><<<gridDim, blockDim, sm>>>(*input, error, gradient_weights);
		break;
	case 3:
		convolution_gradient_weights_reduction<3><<<gridDim, blockDim, sm>>>(*input, error, gradient_weights);
		break;
	case 5:
		convolution_gradient_weights_reduction<5><<<gridDim, blockDim, sm>>>(*input, error, gradient_weights);
		break;
	default:
		assert(false);
	}

	// TODO: Optimierung von Folie 18 noch nicht für den bias-Kernel gemacht.
	gridDim.x *= 2;
	gridDim.y *= 2;
	gridDim.z = output_channels;
	convolution_gradient_bias_reduction<<<gridDim, blockDim, sm>>>(error, gradient_bias);

	optimizer->update(weights, bias, gradient_weights, gradient_bias);
	delete this->input;
	this->input = nullptr;
	return next_error;
}

