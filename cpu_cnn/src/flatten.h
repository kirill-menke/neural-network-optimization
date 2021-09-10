#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

/* This class is used to flatten the spatial dimensions: (B, C, H, W) -> (B, C x H x W, 1, 1).
   The flattened tensor can then be used for a 1x1 convolution (equivalent to Fully-Connected layer) */
class Flatten {
	Eigen::DSizes<Eigen::DenseIndex, 4> input_dims;

public:
	std::shared_ptr<Eigen::Tensor<float, 4>> forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor);
	std::shared_ptr<Eigen::Tensor<float, 4>> backward(std::shared_ptr<Eigen::Tensor<float, 4> const> error_tensor);
};


/* This class is used to flatten the rank dimensions: (B, C, 1, 1) -> (B, C).
   The flattened tensor can then be passed to softmax */
class FlattenRank {
	Eigen::DSizes<Eigen::DenseIndex, 4> input_dims;

public:
	std::shared_ptr<Eigen::Tensor<float, 2>> forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor);
	std::shared_ptr<Eigen::Tensor<float, 4>> backward(std::shared_ptr<Eigen::Tensor<float, 2> const> error_tensor);
};
