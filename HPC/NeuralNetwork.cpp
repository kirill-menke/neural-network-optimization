#include <iostream>
#include <algorithm>

#include "NeuralNetwork.h"

#if 0
float NeuralNetwork::forward(std::shared_ptr<Eigen::Tensor<float, 4>> images, std::shared_ptr<Eigen::Tensor<float, 2>> label_tensor) {

	for (const auto& layer: layers) {
		images = layer->forward(images);
	}
	
	return loss_layer.forward(images, label_tensor);
}

void NeuralNetwork::backward(std::shared_ptr<Eigen::Tensor<float, 2>> label_tensor) {
	
	auto error_tensor = loss_layer.backward(label_tensor);

	for (auto it = layers.rbegin(); it != layers.rend(); ++it)
	{
		error_tensor = (*it)->backward(error_tensor);
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
		auto batch = batch_loader.loadBatch(batch_size);

		float loss = forward(batch.second, batch.first);

		std::cout << "(" << i << ")" << " Loss: " << loss << std::endl;
		
		losses.push_back(loss);
		backward(batch.first);
	}
}

std::shared_ptr<Eigen::Tensor<float, 4> const> NeuralNetwork::test(std::shared_ptr<Eigen::Tensor<float, 4> const> test_data) {
	for (const auto& layer: layers) {
		test_data = layer->forward(test_data);
	}
	return test_data;
}
#endif