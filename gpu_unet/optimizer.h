#pragma once
#include <tuple>
#include <limits>
#include "tensor.h"

class Optimizer {
public:
	Optimizer(float learning_rate):
		learning_rate(learning_rate) {}

	virtual ~Optimizer() {}

	float learning_rate;
	virtual void update(Tensor<float, 4> &weights, Tensor<float, 1> &bias, Tensor<float, 4> &gradient_weights, Tensor<float, 1> &gradient_bias) = 0;
};

class Sgd: public Optimizer {
public:
	Sgd(float learning_rate):
		Optimizer(learning_rate) {}

	void update(Tensor<float, 4> &weights, Tensor<float, 1> &bias, Tensor<float, 4> &gradient_weights, Tensor<float, 1> &gradient_bias);
};


class SgdWithMomentum : public Optimizer {
	float momentum_rate;
	Tensor<float, 4> v_weight;
	Tensor<float, 1> v_bias;

public:
	SgdWithMomentum(float learning_rate, float momentum_rate, std::tuple<int, int, int, int> weight_dims) :
		Optimizer(learning_rate), momentum_rate(momentum_rate),
		v_weight({ std::get<0>(weight_dims), std::get<1>(weight_dims), std::get<2>(weight_dims), std::get<3>(weight_dims) }),
		v_bias({ std::get<0>(weight_dims) }) {

		v_weight.setZero();
		v_bias.setZero();
	}

	void update(Tensor<float, 4> &weights, Tensor<float, 1> &bias,
		Tensor<float, 4> &gradient_tensor, Tensor<float, 1> &gradient_bias);
};


class Adam : public Optimizer {
	float mu;
	float rho;
	int iter;
	Tensor<float, 4> v_weight, r_weight;
	Tensor<float, 1> v_bias, r_bias;

public:
	Adam(float learning_rate, float mu, float rho, std::tuple<int, int, int, int> weight_dims) :
		Optimizer(learning_rate), mu(mu), rho(rho), iter(1),
		v_weight({ std::get<0>(weight_dims), std::get<1>(weight_dims), std::get<2>(weight_dims), std::get<3>(weight_dims) }),
		r_weight({ std::get<0>(weight_dims), std::get<1>(weight_dims), std::get<2>(weight_dims), std::get<3>(weight_dims) }),
		v_bias({ std::get<0>(weight_dims) }), r_bias({ std::get<0>(weight_dims) }) {

		v_weight.setZero();
		r_weight.setZero();

		v_bias.setZero();
		r_bias.setZero();
	}

	void update(Tensor<float, 4>& weights, Tensor<float, 1>& bias,
		Tensor<float, 4>& gradient_tensor, Tensor<float, 1>& gradient_bias);
};



