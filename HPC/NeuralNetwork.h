#pragma once
#include <vector>
#include <memory>

#include <unsupported/Eigen/CXX11/Tensor>

#include "Layer.h"
#include "Optimizer.h"
#include "Initializer.h"
#include "Loss.h"
#include "Helper.h"

class NeuralNetwork {
	std::shared_ptr<Optimizer> optimizer;
	std::shared_ptr<Initializer> initializer;
	CrossEntropyLoss loss_layer;

	std::vector<float> losses;
	std::vector<std::unique_ptr<Layer>> layers;

	MNISTLoader batch_loader;
	int batch_size;
  

public:
	
	NeuralNetwork(std::shared_ptr<Optimizer> optimizer, std::shared_ptr<Initializer> initializer, int batch_size) :
		optimizer(optimizer), initializer(initializer), batch_size(batch_size), batch_loader("../train.txt") {};

	float forward(std::shared_ptr<Eigen::Tensor<float, 4>> images, std::shared_ptr<Eigen::Tensor<float, 2>> labels);

	void backward(std::shared_ptr<Eigen::Tensor<float, 2>> labels);

	void appendLayer(std::unique_ptr<Layer> layer);

	void train(int iterations);

	std::shared_ptr<Eigen::Tensor<float, 4> const> test(std::shared_ptr<Eigen::Tensor<float, 4> const> test_data);

};
