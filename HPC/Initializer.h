#pragma once
#include <tuple>

#include <unsupported/Eigen/CXX11/Tensor>

class Initializer {
public:
	virtual void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 4>& bias) = 0;
};

class UniformRandom : public Initializer {
public:
	void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 4>& bias) {
		// Initialize weights	 
		auto weights_dims = weights.dimensions();
		std::mt19937_64 rng(0);
		std::uniform_real_distribution<float> unif(0, 1);

		for (int i = 0; i < weights_dims[0]; i++) {
			for (int j = 0; j < weights_dims[1]; j++) {
				for (int x = 0; x < weights_dims[2]; x++) {
					for (int y = 0; y < weights_dims[3]; y++) {
						weights(i, j, x, y) = unif(rng);
					}
				}
			}
		}

		// Initialize bias
		for (int i = 0; i < bias.dimension(3); i++) {
			bias(0, 0, 0, i) = unif(rng);
		}
	}
};

class Constant : public Initializer {
	float value;

public: 

	Constant(float value) : value(value) {}

	void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 4>& bias) {
		// Initialize weights	 
		auto weights_dims = weights.dimensions();

		for (int i = 0; i < weights_dims[0]; i++) {
			for (int j = 0; j < weights_dims[1]; j++) {
				for (int x = 0; x < weights_dims[2]; x++) {
					for (int y = 0; y < weights_dims[3]; y++) {
						weights(i, j, x, y) = value;
					}
				}
			}
		}

		// Initialize bias
		for (int i = 0; i < bias.dimension(3); i++) {
			bias(0, 0, 0, i) = value;
		}
	}
};