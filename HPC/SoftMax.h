#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include <memory>

#include "Layer.h"

class SoftMax : public Layer{
public:
	std::shared_ptr<Eigen::Tensor<float, 4>> forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor);
	std::shared_ptr<Eigen::Tensor<float, 4>> backward(std::shared_ptr<Eigen::Tensor<float, 4> const> error_tensor);

private:
	std::shared_ptr<Eigen::Tensor<float, 4>> output_tensor;
};
