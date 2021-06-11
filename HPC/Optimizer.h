#pragma once
#include <limits>
#include <math.h>
#include <unsupported/Eigen/CXX11/Tensor>

class Optimizer {

public:
	virtual Eigen::Tensor<float, 4> calculateUpdate(Eigen::Tensor<float, 4> weight_tensor, Eigen::Tensor<float, 4> gradient_tensor) = 0;
};

class Sgd : public Optimizer {
	float learning_rate;

public:

	Sgd(float learning_rate) : learning_rate(learning_rate) {}

	Eigen::Tensor<float, 4> calculateUpdate(Eigen::Tensor<float, 4> weight_tensor, Eigen::Tensor<float, 4> gradient_tensor) {
		return weight_tensor - learning_rate * gradient_tensor;
	}
};


class SgdWithMomentum : public Optimizer {
	float learning_rate;
	float momentum_rate;
	Eigen::Tensor<float, 4> v;
	
public:
	SgdWithMomentum(float learning_rate, float momentum_rate, std::tuple<int, int, int, int> kernel_dims) : learning_rate(learning_rate), momentum_rate(momentum_rate) {
		v = Eigen::Tensor<float, 4>(std::get<0>(kernel_dims), std::get<1>(kernel_dims), std::get<2>(kernel_dims), std::get<3>(kernel_dims));
		v.setConstant(0.0f);
	}

	Eigen::Tensor<float, 4> calculateUpdate(Eigen::Tensor<float, 4> weight_tensor, Eigen::Tensor<float, 4> gradient_tensor) {
		v = momentum_rate * v - learning_rate * gradient_tensor;
		return weight_tensor + v;
	}
};


class Adam : public Optimizer {
	float learning_rate;
	float mu;
	float rho;
	Eigen::Tensor<float, 4> v;
	Eigen::Tensor<float, 4> r;
	Eigen::Tensor<float, 4> eps;
	int iter = 1;

	Adam(float learning_rate, float mu, float rho, std::tuple<int, int, int, int> kernel_dims) : learning_rate(learning_rate), mu(mu), rho(rho) {
		v = Eigen::Tensor<float, 4>(std::get<0>(kernel_dims), std::get<1>(kernel_dims), std::get<2>(kernel_dims), std::get<3>(kernel_dims));
		r = Eigen::Tensor<float, 4>(std::get<0>(kernel_dims), std::get<1>(kernel_dims), std::get<2>(kernel_dims), std::get<3>(kernel_dims));
		eps = Eigen::Tensor<float, 4>(std::get<0>(kernel_dims), std::get<1>(kernel_dims), std::get<2>(kernel_dims), std::get<3>(kernel_dims));
		v.setConstant(0.0f);
		r.setConstant(0.0f);
		eps.setConstant(std::numeric_limits<float>::epsilon());
	}

	Eigen::Tensor<float, 4> calculateUpdate(Eigen::Tensor<float, 4> weight_tensor, Eigen::Tensor<float, 4> gradient_tensor) {
		v = mu * v + (1 - mu) * gradient_tensor;
		r = rho * r + (1 - rho) * gradient_tensor.pow(2);
		Eigen::Tensor<float, 4> v_corr = v * static_cast<float>((1 / std::pow(1 - mu, iter)));
		Eigen::Tensor<float, 4> r_corr = r * static_cast<float>((1 / std::pow(1 - rho, iter)));
		
		iter++;

		return weight_tensor - learning_rate * v_corr / (r_corr.sqrt() + eps);
	}
};