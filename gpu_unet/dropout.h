#pragma once
#include <memory>
#include <random>

#include "tensor.h"


class Dropout {

	float probability;
	bool testing_phase;
	std::shared_ptr<Tensor<float, 4>> mask;

public:
	Dropout(float probability = 0.8) : probability(probability), testing_phase(false) {};

	Tensor<float, 4> forward(Tensor<float, 4> const & input);
	Tensor<float, 4> backward(Tensor<float, 4> const & error_tensor);

	void setTestingPhase(bool test) {
		testing_phase = test;
	}
};
