#include "flatten.h"

std::shared_ptr<Eigen::Tensor<float, 4>> 
Flatten::forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor) {
	input_dims = input_tensor->dimensions();
	Eigen::array<Eigen::DenseIndex, 4> new_dims{ {input_dims[0], input_dims[1] * input_dims[2] * input_dims[3], 1, 1} };
	return std::make_shared<Eigen::Tensor<float, 4>>(input_tensor->reshape(new_dims));
}


std::shared_ptr<Eigen::Tensor<float, 4>> 
Flatten::backward(std::shared_ptr<Eigen::Tensor<float, 4> const> error_tensor) {
	return std::make_shared<Eigen::Tensor<float, 4>>(error_tensor->reshape(input_dims));
}


std::shared_ptr<Eigen::Tensor<float, 2>> 
FlattenRank::forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor) {
	input_dims = input_tensor->dimensions();
	Eigen::array<Eigen::DenseIndex, 2> new_dims({ input_dims[0], input_dims[1] * input_dims[2] * input_dims[3] });
	return std::make_shared<Eigen::Tensor<float, 2>>(input_tensor->reshape(new_dims));
}


std::shared_ptr<Eigen::Tensor<float, 4>> 
FlattenRank::backward(std::shared_ptr<Eigen::Tensor<float, 2> const> error_tensor) {
	return std::make_shared<Eigen::Tensor<float, 4>>(error_tensor->reshape(input_dims));
}