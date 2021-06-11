#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

#include "Optimizer.h"
#include "Initializer.h"

class Layer {

public:
	bool trainable = false;
	
	virtual Eigen::Tensor<float, 4> forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor) = 0;
	virtual Eigen::Tensor<float, 4> backward(std::shared_ptr<Eigen::Tensor<float, 4> const> error_tensor) = 0;
	virtual void setOptimizer(std::shared_ptr<Optimizer> optimizer) {};
	virtual void setInitializer(std::shared_ptr<Initializer> initializer) {};
};

