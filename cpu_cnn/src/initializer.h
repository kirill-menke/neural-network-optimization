#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

class Initializer {

public:
	virtual void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 1>& bias) = 0;
};


class UniformRandom : public Initializer {
	void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 1>& bias);
};


class Constant : public Initializer {
	float value;

	Constant(float value);
	void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 1>& bias);
};


class He : public Initializer {
	void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 1>& bias);
};


class Xavier : public Initializer {
	void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 1>& bias);
};