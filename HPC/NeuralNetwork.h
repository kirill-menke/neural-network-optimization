#pragma once
#include <vector>

#include "Layer.h"
#include "Optimizer.h"
#include "Initializer.h"

class NeuralNetwork {
	Optimizer& optimizer;
	Initializer& initializer;
	std::vector<float> losses;
	std::vector<Layer> layers;
	Layer& loss_layer;
	Layer& data_layer;


public:
	
	NeuralNetwork(Optimizer& optimizer, Initializer& initializer, Layer& loss_layer, Layer& data_layer) :
		optimizer(optimizer), initializer(initializer), loss_layer(loss_layer), data_layer(data_layer) {}

	float forward();

	void backward();

	void appendLayer(Layer& layer);

	void train();

	std::vector<float> test();

};