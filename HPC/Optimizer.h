#pragma once
#include <vector>

class Optimizer {
	virtual std::vector<float> calculateUpdate(std::vector<float> weight_tensor, std::vector<float> gradient_tensor) = 0;
};

class Sgd : public Optimizer {
	float learning_rate;

public:

	Sgd(float learning_rate) : learning_rate(learning_rate) {}

	std::vector<float> calculateUpdate(std::vector<float> weight_tensor, std::vector<float> gradient_tensor) {
		return std::vector<float>();
	}
};

class SgdWithMomentum : Optimizer {
	 
};

class Adam : Optimizer {

};