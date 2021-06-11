#include <iostream>
#include <memory>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "NeuralNetwork.h"
#include "Optimizer.h"
#include "Initializer.h"
#include "Pooling.h"
#include "FullyConnected.h"
#include "Flatten.h"
#include "ReLU.h"
#include "SoftMax.h"
#include "Loss.h"
#include "Layer.h"
#include "Conv.h"
#include "Helper.h"

int main() {
	// Parse image data, construct network, train and test network ...

	int iterations = 200;
	float learning_rate = 0.01f;

	auto train_data = std::make_shared<Eigen::Tensor<float, 4>>(3, 3, 3, 3);
	std::shared_ptr<std::vector<int>> train_labels;
	auto test_data = std::make_shared<Eigen::Tensor<float, 4>>(3, 3, 3, 3);
	std::shared_ptr<std::vector<int>> test_labels;

	NeuralNetwork net(std::make_unique<Sgd>(learning_rate), std::make_unique<UniformRandom>(), train_data, train_labels);
	
	// Construct CNN
	net.appendLayer(std::make_unique<Conv>(4, 3, 3, 1));		// kernels, channels, filter_size, stride
	//net.appendLayer(std::make_unique<ReLU>());
	//net.appendLayer(std::make_unique<Pooling>(2, 2));			// shape, stride 
	//net.appendLayer(std::make_unique<Flatten>());
	//net.appendLayer(std::make_unique<FullyConnected>(0, 0));	// input_dim, output_dim
	//net.appendLayer(std::make_unique<ReLU>());
	//net.appendLayer(std::make_unique<FullyConnected>(0, 0));
	//net.appendLayer(std::make_unique<SoftMax>());
	// net.appendLayer(std::make_unique<CrossEntropyLoss>());

	// Train network
	net.train(iterations);
	
	// Test Network
	Eigen::Tensor<float, 4> result = net.test(test_data);

	// Calculate accuracy ...

	return 0;
}




