#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

#include "Layer.h"

class SoftMax : public Layer {

	Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4> const & input_tensor) {
		return Eigen::Tensor<float, 4>();
	}

	Eigen::Tensor<float, 4> backward(Eigen::Tensor<float, 4> const & error_tensor) {
		return Eigen::Tensor<float, 4>();
	}
};