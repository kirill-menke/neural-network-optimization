#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include "Layer.h"

class FullyConnected : public Layer {
	int input_size;
	int output_size;

	Eigen::Tensor<float, 4> weights;

public:
	FullyConnected(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
		trainable = true;

		weights = Eigen::Tensor<float, 4>(1, 1, input_size + 1, output_size);
	}

	Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& input_tensor) {
		return Eigen::Tensor<float, 4>();
	}

	Eigen::Tensor<float, 4> backward(Eigen::Tensor<float, 4>& error_tensor) {
		return Eigen::Tensor<float, 4>();
	}

};