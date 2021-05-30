#pragma once
#include <iostream>
#include <tuple>
#include <random>
#include <math.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Initializer.h"
#include "Helper.h"

class Layer {

public:
	bool trainable = false;

	virtual Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4> input_tensor) = 0;
	virtual Eigen::Tensor<float, 4> backward(Eigen::Tensor<float, 4> error_tensor) = 0;
	virtual void initialize(Initializer& initializer) = 0;
};

class Conv : public Layer {
	Eigen::Tensor<float, 4> weights;
	Eigen::VectorXf bias;

	int stride;
	int num_kernels;
	int channels;
	int filter_size;

	Eigen::Tensor<float, 4> input_tensor;

public:

	Conv(int num_kernels, int channels, int filter_size, int stride):
		num_kernels(num_kernels), channels(channels), filter_size(filter_size), stride(stride) {
		trainable = true;

		weights = Eigen::Tensor<float, 4>(num_kernels, channels, filter_size, filter_size);
		bias = Eigen::VectorXf(num_kernels);

		// Just for debugging: Set constant values
		weights.setConstant(1);
		bias.setConstant(0);

		// Initialize weights and bias with uniform distribution
		// 
		//std::mt19937_64 rng(0);
		//std::uniform_real_distribution<float> unif(0, 1);

		//for (int i = 0; i < num_kernels; i++) {
		//	for (int j = 0; j < channels; j++) {
		//		for (int x = 0; x < filter_size; x++) {
		//			for (int y = 0; y < filter_size; y++) {
		//				weights(i, j, x, y) = unif(rng);
		//			}
		//		}
		//	}
		//}
	}

	Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4> input_tensor) {
		this->input_tensor = input_tensor;
		auto input_dims = input_tensor.dimensions();

		Eigen::array<std::pair<int, int>, 4> paddings;
		int spatial_pad = static_cast<int>(filter_size / 2);
		paddings[0] = std::make_pair(0, 0);
		paddings[1] = std::make_pair(0, 0);
		paddings[2] = std::make_pair(spatial_pad, spatial_pad);
		paddings[3] = std::make_pair(spatial_pad, spatial_pad);
		Eigen::Tensor<float, 4> padded_input = input_tensor.pad(paddings);

		printTensor(padded_input);

		Eigen::Tensor<float, 4> feature_maps(input_dims[0], num_kernels, input_dims[2], input_dims[3]);
		correlate3D(padded_input, feature_maps);
		
		//TODO: Implement striding directly in correlate3D
		Eigen::array<Eigen::DenseIndex, 4> strides({1, 1, stride, stride});
		Eigen::Tensor<float, 4> strided_feature_maps = feature_maps.stride(strides);
		
		return strided_feature_maps;
	}


	Eigen::Tensor<float, 4> backward(Eigen::Tensor<float, 4> error_tensor) {
		return Eigen::Tensor<float, 4>();
	}


	void initialize(Initializer& initializer) {

	}


private:

	void correlate3D(const Eigen::Tensor<float, 4>& input_tensor, Eigen::Tensor<float, 4>& feature_maps) {
		auto output_dims = feature_maps.dimensions();
		feature_maps.setZero();
		int spatial_pad = static_cast<int>(filter_size / 2);

		for (int i = 0; i < output_dims[0]; i++) {
			for (int j = 0; j < output_dims[1]; j++) {
				for (int x = 0; x < output_dims[2]; x++) {
					for (int y = 0; y < output_dims[3]; y++) {
						for (int c = 0; c < channels; c++) {
							for (int k = 0; k < filter_size; k++) {
								for (int l = 0; l < filter_size; l++) {
									feature_maps(i, j, x, y) += input_tensor(i, c, x + k, y + l) * weights(j, c, k, l);
								} 
							}
						}
						feature_maps(i, j, x, y) += bias(j);
					}
				}
			}
		}
	}


	void convolve(const Eigen::Tensor<float, 4>& input_tensor, Eigen::Tensor<float, 4>& feature_maps) {
		// Flip the kernel and then use correlate
		correlate3D(input_tensor, feature_maps);
	}

};