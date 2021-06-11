#pragma once
#include <vector>
#include <memory>

#include <unsupported/Eigen/CXX11/Tensor>

#include "Layer.h"
#include "Optimizer.h"
#include "Initializer.h"

class NeuralNetwork {
	std::shared_ptr<Optimizer> optimizer;
	std::shared_ptr<Initializer> initializer;
	std::vector<float> losses;
	std::vector<std::unique_ptr<Layer>> layers;

	std::shared_ptr<Eigen::Tensor<float, 4> const> train_data;
	std::shared_ptr<std::vector<int> const> train_labels;
  

public:
	
	NeuralNetwork(std::shared_ptr<Optimizer> optimizer, std::shared_ptr<Initializer> initializer, 
		std::shared_ptr<Eigen::Tensor<float, 4> const> train_data, std::shared_ptr<std::vector<int> const> train_labels) :
		optimizer(optimizer), initializer(initializer), train_data(train_data), train_labels(train_labels){}

	float forward();

	void backward();

	void appendLayer(std::unique_ptr<Layer> layer);

	void train(int iterations);

	Eigen::Tensor<float, 4> test(std::shared_ptr<Eigen::Tensor<float, 4> const> test_data);

};
