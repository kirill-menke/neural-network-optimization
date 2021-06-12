#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

#include "Layer.h"

class Flatten : public Layer {

	Eigen::DSizes<Eigen::DenseIndex, 4> input_dims;

	std::shared_ptr<Eigen::Tensor<float, 4>> forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor) {
		input_dims = input_tensor->dimensions();
		Eigen::array<Eigen::DenseIndex, 4> three_dims{ {input_dims[0], input_dims[1] * input_dims[2] * input_dims[3], 1, 1} };
		return std::make_shared<Eigen::Tensor<float, 4>>(input_tensor->reshape(three_dims));
	}

	std::shared_ptr<Eigen::Tensor<float, 4>> backward(std::shared_ptr<Eigen::Tensor<float, 4> const> error_tensor) {
		return std::make_shared<Eigen::Tensor<float, 4>>(error_tensor->reshape(input_dims));
	}
};