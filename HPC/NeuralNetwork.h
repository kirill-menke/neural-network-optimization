#pragma once
#include <vector>
#include <memory>

#include <unsupported/Eigen/CXX11/Tensor>

#include "Layer.h"
#include "Optimizer.h"
#include "Initializer.h"

class NeuralNetwork {
	std::unique_ptr<Optimizer> optimizer;
	std::unique_ptr<Initializer> initializer;
	std::vector<float> losses;
	std::vector<std::unique_ptr<Layer>> layers;

	Eigen::Tensor<float, 4> train_data;
	std::vector<int> train_labels;
  

public:
	
	NeuralNetwork(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<Initializer> initializer, Eigen::Tensor<float, 4> train_data, std::vector<int> train_labels) :
		optimizer(std::move(optimizer)), initializer(std::move(initializer)), train_data(train_data), train_labels(train_labels){}

	float forward();

	void backward();

	void appendLayer(std::unique_ptr<Layer> layer);

	void train(int iterations);

	Eigen::Tensor<float, 4> test(Eigen::Tensor<float, 4> test_data);

};
