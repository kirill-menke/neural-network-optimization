#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

class CrossEntropyLoss {
public:
	/*
	 * The label_tensor should be one-hot-encoded!
	 */
	float forward(
		std::shared_ptr<Eigen::Tensor<float, 2>> input_tensor,
		std::shared_ptr<Eigen::Tensor<float, 2>> label_tensor);
	std::shared_ptr<Eigen::Tensor<float, 2>> backward(
		std::shared_ptr<Eigen::Tensor<float, 2>> label_tensor);

private:
	std::shared_ptr<Eigen::Tensor<float, 2>> input_tensor;
};
