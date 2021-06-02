#pragma once
#include <iostream>
#include <tuple>
#include <random>
#include <math.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Layer.h"
#include "Helper.h"

class Conv : public Layer {
	Eigen::Tensor<float, 4> weights;
	Eigen::Tensor<float, 4> r_weights;
	Eigen::VectorXf bias;

	int stride;
	int num_kernels;
	int channels;
	int filter_size;

	Eigen::Tensor<float, 4>* input_tensor;

public:

	Conv(int num_kernels, int channels, int filter_size, int stride) :
		num_kernels(num_kernels), channels(channels), filter_size(filter_size), stride(stride) {
		trainable = true;

		weights = Eigen::Tensor<float, 4>(num_kernels, channels, filter_size, filter_size);
		bias = Eigen::VectorXf(num_kernels);

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
			bias(i) = unif(rng);
		}

		// Reverse kernels for convolution
		Eigen::array<bool, 4> reverse({ false, false, true, true });
		r_weights = weights.reverse(reverse);
	}

	Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& input_tensor) {
		this->input_tensor = &input_tensor;
		auto input_dims = input_tensor.dimensions();
		int spatial_pad = static_cast<int>(filter_size / 2);
		int out_x = static_cast<int>((input_dims[2] - filter_size + 2 * spatial_pad) / stride + 1);
		int out_y = static_cast<int>((input_dims[3] - filter_size + 2 * spatial_pad) / stride + 1);

		Eigen::array<std::pair<int, int>, 4> paddings;
		paddings[0] = std::make_pair(0, 0);
		paddings[1] = std::make_pair(0, 0);
		paddings[2] = std::make_pair(spatial_pad, spatial_pad);
		paddings[3] = std::make_pair(spatial_pad, spatial_pad);
		Eigen::Tensor<float, 4> padded_input = input_tensor.pad(paddings);

		Eigen::Tensor<float, 4> feature_maps(input_dims[0], num_kernels, out_x, out_y);
		correlate(padded_input, feature_maps);

		return feature_maps;
	}


	Eigen::Tensor<float, 4> backward(Eigen::Tensor<float, 4>& error_tensor) {
		auto input_dims = input_tensor->dimensions();
		auto error_dims = error_tensor.dimensions();

		// Next error tensor: Gradient w.r.t the different channels of the input
		Eigen::Tensor<float, 4> upsampled_error_tensor(input_dims[0], num_kernels, input_dims[2], input_dims[3]);

		for (int i = 0; i < error_dims[0]; i++) {
			for (int j = 0; j < error_dims[1]; j++) {
				for (int x = 0, s_x = 0; x < error_dims[2]; x++, s_x+=stride) {
					for (int y = 0, s_y = 0; y < error_dims[3]; y++, s_y+=stride) {
						upsampled_error_tensor(i, j, s_x, s_y) = error_tensor(i, j, x, y);
					}
				}
			}
		}

		// Reverse Tensor
		Eigen::array<bool, 4> reverse({ false, true, false, false });
		Eigen::Tensor<int, 4> reversed_error_tensor = upsampled_error_tensor.reverse(reverse);


		return Eigen::Tensor<float, 4>();
	}


private:
	void convolve(const Eigen::Tensor<float, 4>& input_tensor, Eigen::Tensor<float, 4>& feature_maps) {
		correlate3D(input_tensor, feature_maps, r_weights);
	}

	void correlate(const Eigen::Tensor<float, 4>& input_tensor, Eigen::Tensor<float, 4>& feature_maps) {
		correlate3D(input_tensor, feature_maps, weights);
	}

	void correlate3D(const Eigen::Tensor<float, 4>& input_tensor, Eigen::Tensor<float, 4>& feature_maps, Eigen::Tensor<float, 4>& kernels) {
		auto output_dims = feature_maps.dimensions();
		int spatial_pad = static_cast<int>(filter_size / 2);
		feature_maps.setZero();

		for (int i = 0; i < output_dims[0]; i++) {
			for (int j = 0; j < output_dims[1]; j++) {
				for (int x = 0, s_x = 0; x < output_dims[2]; x++, s_x += stride) {
					for (int y = 0, s_y = 0; y < output_dims[3]; y++, s_y += stride) {
						for (int c = 0; c < channels; c++) {
							for (int k = 0; k < filter_size; k++) {
								for (int l = 0; l < filter_size; l++) {
									feature_maps(i, j, x, y) += input_tensor(i, c, s_x + k, s_y + l) * weights(j, c, k, l);
								}
							}
						}
						feature_maps(i, j, x, y) += bias(j);
					}
				}
			}
		}
	}

};
