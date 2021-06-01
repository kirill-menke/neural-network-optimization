#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

#include "Initializer.h"

class Layer {

public:
	bool trainable = false;
	
	virtual Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& input_tensor) = 0;
	virtual Eigen::Tensor<float, 4> backward(Eigen::Tensor<float, 4>& error_tensor) = 0;
};

