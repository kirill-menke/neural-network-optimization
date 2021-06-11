#include <iostream>

#include "NeuralNetwork.h"


float NeuralNetwork::forward() {
	// TODO: Take a small batch of training data
	auto input = train_data;

	for (const auto& layer: layers) {
		input = std::make_shared<Eigen::Tensor<float, 4>>(layer->forward(input)); 
	}
	// TODO: Pass last result to loss layer and return loss

	return 0.0;
}

void NeuralNetwork::backward() {
	// TODO: Call backward on loss layer to get last error_tensor
	auto error_tensor = std::make_shared<Eigen::Tensor<float, 4>>(3, 3, 3, 3);

	for (const auto& layer : layers) {
		error_tensor = std::make_shared<Eigen::Tensor<float, 4>>(layer->backward(error_tensor));
	}
}

void NeuralNetwork::appendLayer(std::unique_ptr<Layer> layer) {
	if (layer->trainable) {
		layer->setOptimizer(optimizer);
		layer->setInitializer(initializer);
	}

	layers.push_back(std::move(layer));

}

void NeuralNetwork::train(int iterations) {
	for (int i = 0; i < iterations; i++) {
		losses.push_back(forward());
		backward();
	}
}

Eigen::Tensor<float, 4> NeuralNetwork::test(std::shared_ptr<Eigen::Tensor<float, 4> const> test_data) {
	for (const auto& layer: layers) {
		test_data = std::make_shared<Eigen::Tensor<float, 4>>(layer->forward(test_data));
	}
	return *test_data;
}