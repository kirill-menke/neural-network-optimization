#include <iostream>
#include <memory>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "NeuralNetwork.h"
#include "Optimizer.h"
#include "Initializer.h"
#include "MaxPool.h"
#include "FullyConnected.h"
#include "Flatten.h"
#include "ReLU.h"
#include "SoftMax.h"
#include "Loss.h"
#include "Layer.h"
#include "Conv.h"
#include "Helper.h"

int main() {

	int iterations = 200;
	float learning_rate = 0.01f;

	NeuralNetwork net(std::make_unique<Sgd>(learning_rate), std::make_unique<UniformRandom>(), 10);
	
	// Construct CNN
	net.appendLayer(std::make_unique<Conv>(4, 3, 3, 1));			// kernels, channels, filter_size, stride
	net.appendLayer(std::make_unique<ReLU>());
	net.appendLayer(std::make_unique<MaxPool>(2, 2, 2, 2));			// stride, shape
	net.appendLayer(std::make_unique<Flatten>());
	net.appendLayer(std::make_unique<Conv>(10, 784, 1, 1));
	net.appendLayer(std::make_unique<SoftMax>());

	// Train network
	net.train(iterations);
	
	// Test Network
	// Eigen::Tensor<float, 4> labels = net.test(test_data);

	// Calculate accuracy ...



	return 0;
}




