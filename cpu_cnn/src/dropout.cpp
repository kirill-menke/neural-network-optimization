#include <random>

#include "dropout.h"


Dropout::Dropout(float probability) : probability(probability), testing_phase(false) {};


std::shared_ptr<Eigen::Tensor<float, 4>>
Dropout::forward(std::shared_ptr<Eigen::Tensor<float, 4>> input) {

	if (testing_phase)
		return input;

	auto dims = input->dimensions();

	// Allocate memory once and reuse it in the following forward calls
	if (!mask)
		mask = std::make_unique<Eigen::Tensor<float, 4>>(dims);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> distrib({ 1. - probability, probability });

	for (int b = 0; b < dims[0]; b++) {
		for (int c = 0; c < dims[1]; c++) {
			for (int x = 0; x < dims[2]; x++) {
				for (int y = 0; y < dims[3]; y++) {
					int activation = distrib(gen);
					(*mask)(b, c, x, y) = activation;
					(*input)(b, c, x, y) = ((*input)(b, c, x, y) * activation) / probability;
				}
			}
		}
	}

	return input;
}


std::shared_ptr<Eigen::Tensor<float, 4>>
Dropout::backward(std::shared_ptr<Eigen::Tensor<float, 4>> error_tensor) {
	auto dims = error_tensor->dimensions();

	for (int b = 0; b < dims[0]; b++) {
		for (int c = 0; c < dims[1]; c++) {
			for (int x = 0; x < dims[2]; x++) {
				for (int y = 0; y < dims[3]; y++) {
					(*error_tensor)(b, c, x, y) *= (*mask)(b, c, x, y);
				}
			}
		}
	}

	return error_tensor;
}


void Dropout::setTestingPhase(bool test) {
	testing_phase = test;
}