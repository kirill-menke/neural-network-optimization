#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

#include "Layer.h"

class Flatten : public Layer {

	Eigen::DSizes<Eigen::DenseIndex, 4> input_size;

	Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& input_tensor) {
		input_size = input_tensor.dimensions();
		return Eigen::Tensor<float, 4>();
	}

	Eigen::Tensor<float, 4> backward(Eigen::Tensor<float, 4>& error_tensor) {
		return Eigen::Tensor<float, 4>();
	}
};