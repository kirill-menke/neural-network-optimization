#pragma once
#include <iostream>
#include <tuple>
#include <random>
#include <math.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Layer.h"
#include "Optimizer.h"
#include "Helper.h"

class Conv : public Layer {
	std::shared_ptr<Optimizer> optimizer;
	std::shared_ptr<Initializer> initializer;

	Eigen::Tensor<float, 4> weights;
	Eigen::Tensor<float, 4> gradient_weights;
	Eigen::Tensor<float, 4> bias;
	Eigen::Tensor<float, 4> gradient_bias;

	int stride;
	int num_kernels;
	int channels;
	int filter_size;
	int spatial_pad;

	std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor;
	Eigen::array<std::pair<int, int>, 4> paddings;

public:

	Conv(int num_kernels, int channels, int filter_size, int stride) :
		num_kernels(num_kernels), channels(channels), filter_size(filter_size), stride(stride) {
		trainable = true;

		weights = Eigen::Tensor<float, 4>(num_kernels, channels, filter_size, filter_size);
		gradient_weights = Eigen::Tensor<float, 4>(num_kernels, channels, filter_size, filter_size);
		bias = Eigen::Tensor<float, 4>(1, 1, 1, num_kernels);
		gradient_bias = Eigen::Tensor<float, 4>(1, 1, 1, num_kernels);

		// Initialize weights	 
		std::mt19937_64 rng(0);
		std::uniform_real_distribution<float> unif(0, 1);

		for (int i = 0; i < num_kernels; i++) {
			for (int j = 0; j < channels; j++) {
				for (int x = 0; x < filter_size; x++) {
					for (int y = 0; y < filter_size; y++) {
						weights(i, j, x, y) = unif(rng);
					}
				}
			}
		}

		// Initialize bias
		for (int i = 0; i < num_kernels; i++) {
			bias(0, 0, 0, i) = unif(rng);
		}


		// Create padding so that tensor size remains the same after correlation/ convolution
		spatial_pad = static_cast<int>(filter_size / 2);
		paddings[0] = std::make_pair(0, 0);
		paddings[1] = std::make_pair(0, 0);
		paddings[2] = std::make_pair(spatial_pad, spatial_pad);
		paddings[3] = std::make_pair(spatial_pad, spatial_pad);
	}

	std::shared_ptr<Eigen::Tensor<float, 4>> forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor) {
		this->input_tensor = input_tensor;
		auto input_dims = input_tensor->dimensions();

		Eigen::Tensor<float, 4> padded_input = input_tensor->pad(paddings);

		int out_x = static_cast<int>((input_dims[2] - filter_size + 2 * spatial_pad) / stride + 1);
		int out_y = static_cast<int>((input_dims[3] - filter_size + 2 * spatial_pad) / stride + 1);
		auto feature_maps = std::make_shared<Eigen::Tensor<float, 4>>(input_dims[0], num_kernels, out_x, out_y);
		correlate(padded_input, *feature_maps, stride, true);

		return feature_maps;

	}


	std::shared_ptr<Eigen::Tensor<float, 4>> backward(std::shared_ptr<Eigen::Tensor<float, 4> const> error_tensor) {

		auto input_dims = input_tensor->dimensions();
		auto error_dims = error_tensor->dimensions();

		/* Next error tensor: Gradient w.r.t the different channels of the input */

		// Upsample error tensor if stride was > 1
		Eigen::Tensor<float, 4> upsampled_error_tensor(input_dims[0], num_kernels, input_dims[2], input_dims[3]);

		for (int i = 0; i < error_dims[0]; i++) {
			for (int j = 0; j < error_dims[1]; j++) {
				for (int x = 0, s_x = 0; x < error_dims[2]; x++, s_x+=stride) {
					for (int y = 0, s_y = 0; y < error_dims[3]; y++, s_y+=stride) {
						upsampled_error_tensor(i, j, s_x, s_y) = (*error_tensor)(i, j, x, y);
					}
				}
			}
		}


		Eigen::Tensor<float, 4> const & padded_error_tensor = upsampled_error_tensor.pad(paddings);
		auto feature_maps = std::make_shared<Eigen::Tensor<float, 4>>(input_dims);
		convolve(padded_error_tensor, *feature_maps, 1, false);
		
		/* Gradient w.r.t. the bias and weights*/
		gradient_bias.setZero();
		gradient_weights.setZero();

		for (int i = 0; i < input_dims[0]; i++) {
			for (int j = 0; j < num_kernels; j++) {
				for (int x = 0; x < input_dims[2]; x++) {
					for (int y = 0; y < input_dims[3]; y++) {
						gradient_bias(0, 0, 0, j) += upsampled_error_tensor(i, j, x, y);
					}
				}
			}
		}


		Eigen::Tensor<float, 4> const & padded_input_tensor = input_tensor->pad(paddings);
		Eigen::Tensor<float, 4> kernel_gradient(1, channels, filter_size, filter_size);
		Eigen::array<Eigen::DenseIndex, 4> slice_dim{1, 1, input_dims[2], input_dims[3]};

		for (int i = 0; i < error_dims[0]; i++) {
			for (int j = 0; j < error_dims[1]; j++) {
				Eigen::Tensor<float, 4> const& slice = upsampled_error_tensor.chip(0, 0).chip(j, 0).reshape(slice_dim);
				correlate3D(padded_input_tensor, kernel_gradient, slice, 1, false);
				gradient_weights.chip(j, 0) += kernel_gradient.chip(0, 0);
			}
		}


		// Update weights and bias
		if (optimizer) {
			weights = optimizer->calculateUpdate(weights, gradient_weights);
			bias = optimizer->calculateUpdate(bias, gradient_bias);
		}

		return feature_maps;
	}

	void setWeights(Eigen::Tensor<float, 4> weights) {
		this->weights = weights;
	}

	void setBias(Eigen::Tensor<float, 4> bias) {
		this->bias = bias;
	}

	void setOptimizer(std::shared_ptr<Optimizer> optimizer) {
		this->optimizer = optimizer;
	}

	Eigen::Tensor<float, 4> getWeights() {
		return weights;
	}

	Eigen::Tensor<float, 4> getBias() {
		return bias;
	}

	Eigen::Tensor<float, 4> getGradientWeights() {
		return gradient_weights;
	}

	Eigen::Tensor<float, 4> getGradientBias() {
		return gradient_bias;
	}



private:
	void convolve(const Eigen::Tensor<float, 4>& input, Eigen::Tensor<float, 4>& feature_maps, int stride, bool add_bias) {
		// Swap kernel axes (batch and color channel) and reverse dims
		Eigen::Tensor<float, 4> r_weights = reverseKernelDims();
		correlate3D(input, feature_maps, r_weights, stride, add_bias);
	}

	Eigen::Tensor<float, 4> reverseKernelDims() {
		Eigen::array<int, 4> shuffle({ 1, 0, 2, 3 });
		Eigen::array<bool, 4> reverse({ true, false, true, true });

		Eigen::Tensor<float, 4> const & s_weights = weights.shuffle(shuffle);
		Eigen::Tensor<float, 4> const & r_weights = s_weights.reverse(reverse);

		return r_weights;
	}

	void correlate(const Eigen::Tensor<float, 4>& input, Eigen::Tensor<float, 4>& feature_maps, int stride, bool add_bias) {
		correlate3D(input, feature_maps, weights, stride, add_bias);
	}

	void correlate3D(Eigen::Tensor<float, 4> const & input, Eigen::Tensor<float, 4>& feature_maps, Eigen::Tensor<float, 4> const & kernels, int stride, bool add_bias) {
		auto output_dims = feature_maps.dimensions();
		auto kernel_dims = kernels.dimensions();
		feature_maps.setZero();

		for (int i = 0; i < output_dims[0]; i++) {
			for (int j = 0; j < output_dims[1]; j++) {
				for (int x = 0, s_x = 0; x < output_dims[2]; x++, s_x += stride) {
					for (int y = 0, s_y = 0; y < output_dims[3]; y++, s_y += stride) {
						for (int c = 0; c < kernel_dims[1]; c++) {
							for (int k = 0; k < kernel_dims[2]; k++) {
								for (int l = 0; l < kernel_dims[3]; l++) {
									feature_maps(i, j, x, y) += input(i, c, s_x + k, s_y + l) * kernels(j, c, k, l);
								}
							}
						}
						if (add_bias)
							feature_maps(i, j, x, y) += bias(j);
					}
				}
			}
		}
	}

};
