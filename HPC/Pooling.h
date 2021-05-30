#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

#include "Layer.h"

class Pooling : Layer {
	int shape;

public:
	Pooling(int shape) : shape(shape) {}

	Eigen::Tensor<float, 4> forward() {
		return Eigen::Tensor<float, 4>();
	}

	Eigen::Tensor<float, 4> backward() {
		return Eigen::Tensor<float, 4>();
	}
};