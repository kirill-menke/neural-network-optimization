#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

class Dropout {

	bool testing_phase;
	float probability;
	std::unique_ptr<Eigen::Tensor<float, 4>> mask;

public:

	Dropout(float probability = 0.8);

	/** Set activations with the given probability to zero or otherwise divide them by the probability */
	std::shared_ptr<Eigen::Tensor<float, 4>> forward(std::shared_ptr<Eigen::Tensor<float, 4>> input);

	std::shared_ptr<Eigen::Tensor<float, 4>> backward(std::shared_ptr<Eigen::Tensor<float, 4>> error_tensor);

	/** During testing phase the input_tensor is forwarded without modfication */
	void setTestingPhase(bool test);
	
};