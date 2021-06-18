#pragma once
#include <tuple>
#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>

class Initializer {

public:
	virtual void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 1>& bias) = 0;
};

class UniformRandom : public Initializer {
public:

	void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 1>& bias) {
		// Initialize weights	 
		auto weights_dims = weights.dimensions();
		std::mt19937_64 rng(0);
		std::uniform_real_distribution<float> unif(-0.2, 0.2);

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
		for (int i = 0; i < bias.dimension(0); i++) {
			bias(i) = unif(rng);
		}
	}
};

class Constant : public Initializer {
	float value;

public: 

	Constant(float value) : value(value) {};

	void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 1>& bias) {
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
		for (int i = 0; i < bias.dimension(0); i++) {
			bias(i) = value;
		}
	}
};


class He : public Initializer {
	void initialize(Eigen::Tensor<float, 4>& weights, Eigen::Tensor<float, 1>& bias) {
		// Initialize weights	 
		auto weight_dims = weights.dimensions();
		std::mt19937_64 rng(0);
		std::normal_distribution<float> distribution_weights(0.0, std::sqrtf(2 / static_cast<float>(weight_dims[2] * weight_dims[3])));
		std::normal_distribution<float> distribution_bias(0.0, std::sqrtf(2 / static_cast<float>(weight_dims[0])));

		for (int i = 0; i < weight_dims[0]; i++) {
			for (int j = 0; j < weight_dims[1]; j++) {
				for (int x = 0; x < weight_dims[2]; x++) {
					for (int y = 0; y < weight_dims[3]; y++) {
						weights(i, j, x, y) = distribution_weights(rng);
					}
				}
			}
		}

		// Initialize bias
		for (int i = 0; i < bias.dimension(0); i++) {
			bias(i) = distribution_bias(rng);
		}
	}
};