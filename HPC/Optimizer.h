#pragma once
#include <vector>

class Optimizer {
	virtual std::vector<float> calculateUpdate(std::vector<float> weight_tensor, std::vector<float> gradient_tensor) = 0;
};

class Sgd : Optimizer {
	std::vector<float> calculateUpdate(std::vector<float> weight_tensor, std::vector<float> gradient_tensor) {
		return std::vector<float>();
	}
};

class SgdWithMomentum : Optimizer {

};

class Adam : Optimizer {

};