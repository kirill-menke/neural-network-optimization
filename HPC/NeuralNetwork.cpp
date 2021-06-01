#include <iostream>

#include "NeuralNetwork.h"


float NeuralNetwork::forward() {
	// TODO: Take a small batch of training data
	Eigen::Tensor<float, 4> input;
	std::vector<float> labels; 

	for (const auto& layer: layers) {
		input = layer->forward(input);
	}
	// TODO: Pass last result to loss layer and return loss

	return 0.0;
}

void NeuralNetwork::backward() {
	// TODO: Call backward on loss layer to get last error_tensor
	Eigen::Tensor<float, 4> error_tensor;

	for (const auto& layer : layers) {
		error_tensor = layer->backward(error_tensor);
	}
}

void NeuralNetwork::appendLayer(std::unique_ptr<Layer> layer) {
	layers.push_back(std::move(layer));
}

void NeuralNetwork::train(int iterations) {
	for (int i = 0; i < iterations; i++) {
		losses.push_back(forward());
		backward();
	}
}

Eigen::Tensor<float, 4> NeuralNetwork::test(Eigen::Tensor<float, 4> test_data) {
	for (const auto& layer: layers) {
		test_data = layer->forward(test_data);
	}
	return test_data;
}