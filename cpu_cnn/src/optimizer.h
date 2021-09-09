#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

class Optimizer {

public:
	virtual void calculateUpdate(Eigen::Tensor<float, 4> &weight_tensor, Eigen::Tensor<float, 4> &gradient_tensor,
		Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 1>& gradient_bias) = 0;
};


class Sgd : public Optimizer {
	float learning_rate;

	Sgd(float learning_rate);
	void calculateUpdate(Eigen::Tensor<float, 4>& weight_tensor, Eigen::Tensor<float, 4>& gradient_tensor, 
		Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 1>& gradient_bias);
};


class SgdWithMomentum : public Optimizer {
	float learning_rate;
	float momentum_rate;
	Eigen::Tensor<float, 4> v_weight;
	Eigen::Tensor<float, 1> v_bias;
	
public:
	SgdWithMomentum(float learning_rate, float momentum_rate, std::tuple<int, int, int, int> weight_dims);
	void calculateUpdate(Eigen::Tensor<float, 4>& weight_tensor, Eigen::Tensor<float, 4>& gradient_tensor,
		Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 1>& gradient_bias);
};


class Adam : public Optimizer {
	float learning_rate;
	float mu;
	float rho;
	int iter;
	Eigen::Tensor<float, 4> v_weight, r_weight, eps_weight;
	Eigen::Tensor<float, 1> v_bias, r_bias, eps_bias;

public:
	Adam(float learning_rate, float mu, float rho, std::array<int, 4> kernel_dims);
	void calculateUpdate(Eigen::Tensor<float, 4>& weight_tensor, Eigen::Tensor<float, 4>& gradient_tensor,
		Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 1>& gradient_bias);
};