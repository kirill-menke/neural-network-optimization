#include "conv.h"
#include "initializer.h"
#include "optimizer.h"

Conv::Conv(int num_kernels, int channels, int filter_size, int stride) :
	stride(stride), num_kernels(num_kernels), channels(channels), filter_size(filter_size), optimizer(nullptr),
	weights(num_kernels, channels, filter_size, filter_size), bias(num_kernels),
	gradient_weights(std::make_shared<Eigen::Tensor<float, 4>>(num_kernels, channels, filter_size, filter_size)),
	gradient_bias(std::make_shared<Eigen::Tensor<float, 1>>(num_kernels)) {

	// Initialize weights and bias with uniform distribution by default
	std::mt19937_64 rng(0.);
	std::uniform_real_distribution<float> unif(-0.025, 0.025);

	for (int i = 0; i < num_kernels; i++) {
		for (int j = 0; j < channels; j++) {
			for (int x = 0; x < filter_size; x++) {
				for (int y = 0; y < filter_size; y++) {
					weights(i, j, x, y) = unif(rng);
				}
			}
		}
	}

	for (int i = 0; i < num_kernels; i++) {
		bias(i) = unif(rng);
	}
}


std::shared_ptr<Eigen::Tensor<float, 4>> 
Conv::forward(std::shared_ptr<Eigen::Tensor<float, 4>> input_tensor) {
	// Pad spatial dimension of input_tensor to get the same size after convolution
	this->input_tensor = pad(input_tensor, filter_size / 2, filter_size / 2);
	return convolutionForward(this->input_tensor, input_tensor->dimension(2), input_tensor->dimension(3));
}


std::shared_ptr<Eigen::Tensor<float, 4>> 
Conv::backward(std::shared_ptr<Eigen::Tensor<float, 4>> error_tensor) {
	
	/* Calculate gradient w.r.t. the input (error tensor passed to the previous layer) */

	auto upsampled_error = stride != 1 ? upsample(error_tensor) : error_tensor;
	auto padded_error = pad(upsampled_error, filter_size / 2, filter_size / 2);

	int batch_size = error_tensor->dimension(0),
		width = input_tensor->dimension(2) - 2 * (filter_size / 2),
		height = input_tensor->dimension(3) - 2 * (filter_size / 2);

	auto next_error = convolutionBackward(padded_error, width, height);


	/* Calculate gradient w.r.t. the weights */
	gradient_weights->setZero();
	gradient_bias->setZero();

	for (int b = 0; b < batch_size; b++) {
		for (int i = 0; i < filter_size; i++) {
			for (int j = 0; j < filter_size; j++) {
				for (int k = 0; k < num_kernels; k++) {
					for (int c = 0; c < channels; c++) {
						float err = 0.0;

						for (int x = 0; x < width; x += stride) {
							for (int y = 0; y < height; y += stride) {
								err += (*input_tensor)(b, c, x + i, y + j) * (*upsampled_error)(b, k, x, y);
							}
						}

						(*gradient_weights)(k, c, i, j) = err;
					}
				}
			}
		}
	}


	int output_width = error_tensor->dimension(2), 
		output_height = error_tensor->dimension(3);

	for (int k = 0; k < num_kernels; k++) {
		float err = 0.0;
		for (int b = 0; b < batch_size; b++) {
			for (int x = 0; x < output_width; x++) {
				for (int y = 0; y < output_height; y++) {
					err += (*error_tensor)(b, k, x, y);
				}
			}
		}
		(*gradient_bias)(k) = err;
	}


	/* Update weights and bias using the calculated gradients */
	if (optimizer) {
		optimizer->calculateUpdate(weights, *gradient_weights, bias, *gradient_bias);
	}

	return next_error;
}


std::shared_ptr<Eigen::Tensor<float, 4>>
Conv::pad(std::shared_ptr<Eigen::Tensor<float, 4>> input, int px, int py) {
	int batch_size = input->dimension(0),
		channels = input->dimension(1),
		width = input->dimension(2),
		height = input->dimension(3);

	auto output = std::make_shared<Eigen::Tensor<float, 4>>(
		batch_size, channels, width + 2 * px, height + 2 * py);
	output->setZero();

	for (int b = 0; b < batch_size; b++) {
		for (int c = 0; c < channels; c++) {
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					(*output)(b, c, x + px, y + py) = (*input)(b, c, x, y);
				}
			}
		}
	}

	return output;
}


std::shared_ptr<Eigen::Tensor<float, 4>>
Conv::upsample(std::shared_ptr<Eigen::Tensor<float, 4>> input) {
	int batch_size = input->dimension(0),
		channels = input->dimension(1),
		width = input->dimension(2),
		height = input->dimension(3);

	auto output = std::make_shared<Eigen::Tensor<float, 4>>(
		batch_size, channels, width * stride, height * stride);
	output->setZero();

	for (int b = 0; b < batch_size; b++) {
		for (int c = 0; c < channels; c++) {
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					(*output)(b, c, x * stride, y * stride) = (*input)(b, c, x, y);
				}
			}
		}
	}

	return output;
}

std::shared_ptr<Eigen::Tensor<float, 4>>
Conv::convolutionForward(std::shared_ptr<Eigen::Tensor<float, 4>> input, int output_width, int output_height) {
	int batch_size = input->dimension(0);
	auto output = std::make_shared<Eigen::Tensor<float, 4>>(
		batch_size, num_kernels, output_width, output_height);

	for (int b = 0; b < batch_size; b++) {
		for (int k = 0; k < num_kernels; k++) {
			for (int x = 0; x < output_width; x++) {
				for (int y = 0; y < output_height; y++) {
					float val = 0.0;
					for (int c = 0; c < channels; c++) {
						for (int i = 0; i < filter_size; i++) {
							for (int j = 0; j < filter_size; j++) {
								val += (*input)(b, c, x * stride + i, y * stride + j)
									* weights(k, c, i, j);
							}
						}
					}

					(*output)(b, k, x, y) = val + bias(k);
				}
			}
		}
	}

	return output;
}


std::shared_ptr<Eigen::Tensor<float, 4>>
Conv::convolutionBackward(std::shared_ptr<Eigen::Tensor<float, 4>> input, int output_width, int output_height) {
	int batch_size = input->dimension(0);
	auto output = std::make_shared<Eigen::Tensor<float, 4>>(
		batch_size, channels, output_width, output_height);

	for (int b = 0; b < batch_size; b++) {
		for (int c = 0; c < channels; c++) {
			for (int x = 0; x < output_width; x++) {
				for (int y = 0; y < output_height; y++) {
					float val = 0.0;
					for (int k = 0; k < num_kernels; k++) {
						for (int i = 0; i < filter_size; i++) {
							for (int j = 0; j < filter_size; j++) {
								val += (*input)(b, k, x * stride + i, y * stride + j)
									* weights(k, c, filter_size - i - 1, filter_size - j - 1);
							}
						}
					}

					(*output)(b, c, x, y) = val;
				}
			}
		}
	}

	return output;
}


void Conv::setOptimizer(Optimizer* optimizer) {
	this->optimizer = optimizer;
}

void Conv::setInitializer(Initializer* initializer) {
	initializer->initialize(weights, bias);
}

Eigen::Tensor<float, 4> Conv::getWeights() {
	return weights;
}

Eigen::Tensor<float, 1> Conv::getBias() {
	return bias;
}

Eigen::Tensor<float, 4> Conv::getGradientWeights() {
	return *gradient_weights;
}

Eigen::Tensor<float, 1> Conv::getGradientBias() {
	return *gradient_bias;
}

std::array<int, 4> Conv::getWeightDims() {
	return { (int)weights.dimension(0), (int)weights.dimension(1), (int)weights.dimension(2), (int)weights.dimension(3) };
}

void Conv::setWeights(Eigen::Tensor<float, 4> weights) {
	this->weights = weights;
}

void Conv::setBias(Eigen::Tensor<float, 1> bias) {
	this->bias = bias;
}

