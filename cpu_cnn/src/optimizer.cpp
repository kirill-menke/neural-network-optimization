#include "optimizer.h"


Sgd::Sgd(float learning_rate) : learning_rate(learning_rate) {}

void Sgd::calculateUpdate(Eigen::Tensor<float, 4>& weight_tensor, Eigen::Tensor<float, 4>& gradient_tensor,
	Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 1>& gradient_bias) {
	
	weight_tensor -= learning_rate * gradient_tensor;
	bias -= learning_rate * gradient_bias;
}


SgdWithMomentum::SgdWithMomentum(float learning_rate, float momentum_rate, std::tuple<int, int, int, int> weight_dims) :
	learning_rate(learning_rate), momentum_rate(momentum_rate),
	v_weight(std::get<0>(weight_dims), std::get<1>(weight_dims), std::get<2>(weight_dims), std::get<3>(weight_dims)),
	v_bias(std::get<0>(weight_dims))
{
	v_weight.setZero();
	v_bias.setZero();
}

void SgdWithMomentum::calculateUpdate(Eigen::Tensor<float, 4>& weight_tensor, Eigen::Tensor<float, 4>& gradient_tensor,
	Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 1>& gradient_bias) {
	v_weight = momentum_rate * v_weight - learning_rate * gradient_tensor;
	weight_tensor += v_weight;

	v_bias = momentum_rate * v_bias - learning_rate * gradient_bias;
	bias += v_bias;
}


Adam::Adam(float learning_rate, float mu, float rho, std::array<int, 4> kernel_dims) : learning_rate(learning_rate), mu(mu), rho(rho), iter(1),
v_weight(kernel_dims[0], kernel_dims[1], kernel_dims[2], kernel_dims[3]),
r_weight(kernel_dims[0], kernel_dims[1], kernel_dims[2], kernel_dims[3]),
eps_weight(kernel_dims[0], kernel_dims[1], kernel_dims[2], kernel_dims[3]),
v_bias(kernel_dims[0]), r_bias(kernel_dims[0]), eps_bias(kernel_dims[0])
{

	v_weight.setZero();
	r_weight.setZero();
	eps_weight.setConstant(std::numeric_limits<float>::epsilon());

	v_bias.setZero();
	r_bias.setZero();
	eps_bias.setConstant(std::numeric_limits<float>::epsilon());
}

void Adam::calculateUpdate(Eigen::Tensor<float, 4>& weight_tensor, Eigen::Tensor<float, 4>& gradient_tensor,
	Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 1>& gradient_bias) {

	v_weight = mu * v_weight + (1 - mu) * gradient_tensor;
	r_weight = rho * r_weight + (1 - rho) * gradient_tensor.pow(2);
	Eigen::Tensor<float, 4> v_corr_w = v_weight * static_cast<float>((1 / (1 - std::pow(mu, iter))));
	Eigen::Tensor<float, 4> r_corr_w = r_weight * static_cast<float>((1 / (1 - std::pow(rho, iter))));

	weight_tensor -= learning_rate * v_corr_w / (r_corr_w.sqrt() + eps_weight);


	v_bias = mu * v_bias + (1 - mu) * gradient_bias;
	r_bias = rho * r_bias + (1 - rho) * gradient_bias.pow(2);
	Eigen::Tensor<float, 1> v_corr_b = v_bias * static_cast<float>((1 / (1 - std::pow(mu, iter))));
	Eigen::Tensor<float, 1> r_corr_b = r_bias * static_cast<float>((1 / (1 - std::pow(rho, iter))));

	bias -= learning_rate * v_corr_b / (r_corr_b.sqrt() + eps_bias);

	iter++;
}
