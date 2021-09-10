#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

class MaxPool {
public:
	MaxPool(
		int strideX,
		int strideY,
		int poolWidth,
		int poolHeight
	):
		inputWidth(0),
		inputHeight(0),
		strideX(strideX),
		strideY(strideY),
		poolWidth(poolWidth),
		poolHeight(poolHeight) {}

	std::shared_ptr<Eigen::Tensor<float, 4>> forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor);
	std::shared_ptr<Eigen::Tensor<float, 4>> backward(std::shared_ptr<Eigen::Tensor<float, 4> const> error_tensor);

	int outputWidth() const {
		return (inputWidth - (inputWidth % poolWidth)) / strideX;
	}

	int outputHeight() const {
		return (inputHeight - (inputHeight % poolHeight)) / strideX;
	}

private:
	std::shared_ptr<Eigen::Tensor<float, 4>> output_tensor;
	int inputWidth;
	int inputHeight;
	int strideX;
	int strideY;
	int poolWidth;
	int poolHeight;
	
	std::shared_ptr<Eigen::Tensor<int, 5>> maximas;
};
