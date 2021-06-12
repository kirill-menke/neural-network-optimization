#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include <memory>

class SoftMax {
public:
	std::shared_ptr<Eigen::Tensor<float, 2>> forward(std::shared_ptr<Eigen::Tensor<float, 2> const> input_tensor);
	std::shared_ptr<Eigen::Tensor<float, 2>> backward(std::shared_ptr<Eigen::Tensor<float, 2> const> error_tensor);

private:
	std::shared_ptr<Eigen::Tensor<float, 2>> output_tensor;
};
