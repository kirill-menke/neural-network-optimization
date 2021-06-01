#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

#include "Layer.h"

class Pooling : public Layer {
	int shape;
	int stride;

public:
	Pooling(int shape, int stride) : shape(shape), stride(stride) {}

	Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& input_tensor) {
		return Eigen::Tensor<float, 4>();
	}

	Eigen::Tensor<float, 4> backward(Eigen::Tensor<float, 4>& error_tensor) {
		return Eigen::Tensor<float, 4>();
	}
};