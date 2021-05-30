#pragma once
#include <tuple>

#include <unsupported/Eigen/CXX11/Tensor>

class Initializer {
	virtual Eigen::Tensor<float, 4> initialize(std::tuple<> shape) = 0;
};

class UniformRandom : Initializer {
public:
	Eigen::Tensor<float, 4> initialize(std::tuple<> shape) {
		return Eigen::Tensor<float, 4>();
	}
};