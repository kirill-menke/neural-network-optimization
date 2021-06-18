#include <iostream>
#include <memory>
#include <assert.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

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

class FlattenRank {
	Eigen::DSizes<Eigen::DenseIndex, 4> input_dims;

public:
	std::shared_ptr<Eigen::Tensor<float, 2>> forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor) {
		input_dims = input_tensor->dimensions();
		Eigen::array<Eigen::DenseIndex, 2> new_dims({ input_dims[0], input_dims[1] * input_dims[2] * input_dims[3] });
		return std::make_shared<Eigen::Tensor<float, 2>>(input_tensor->reshape(new_dims));
	}

	std::shared_ptr<Eigen::Tensor<float, 4>> backward(std::shared_ptr<Eigen::Tensor<float, 2> const> error_tensor) {
		return std::make_shared<Eigen::Tensor<float, 4>>(error_tensor->reshape(input_dims));
	}
};

void assertSize(std::shared_ptr<Eigen::Tensor<float, 4>> tensor, int d1, int d2, int d3, int d4) {
	assert(tensor->dimension(0) == d1);
	assert(tensor->dimension(1) == d2);
	assert(tensor->dimension(2) == d3);
	assert(tensor->dimension(3) == d4);
}

void assertSize(std::shared_ptr<Eigen::Tensor<float, 2>> tensor, int d1, int d2) {
	assert(tensor->dimension(0) == d1);
	assert(tensor->dimension(1) == d2);
}

/*
 * Return percentage of correct predictions:
 */
float correctDigits(std::shared_ptr<Eigen::Tensor<float, 2>> truth, std::shared_ptr<Eigen::Tensor<float, 2>> prediction) {
	int batch_size = truth->dimension(0);
	int classes = truth->dimension(1);
	int correct = 0;
	for (int b = 0; b < batch_size; b++) {
		int maxidx = 0;
		for (int c = 0; c < classes; c++) {
			if ((*prediction)(b, c) > (*prediction)(b, maxidx)) {
				maxidx = c;
			}
		}

		if ((*truth)(b, maxidx) > 0.0) {
			correct += 1;
		}
	}

	return (float(correct) / float(batch_size)) * 100.0;
}

int main() {
	int iterations = 10000;
	int batchSize = 64;
	float learning_rate = 0.01;

	MNISTLoader loader("../data/mnist-train.txt");
	loader.loadFullDataset();
	
	// https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist

	/* B x 1 x 28 x 28 */
	Conv conv1(8, 1, 3, 1);
	/* B x 32 x 28 x 28 */
	ReLU reLu1;
	/* B x 32 x 28 x 28 */
	MaxPool maxPool1(2, 2, 2, 2);
	/* B x 32 x 14 x 14 */
	Conv conv2(16, 8, 3, 1);
	/* B x 64 x 14 x 14 */
	ReLU reLu2;
	/* B x 64 x 14 x 14 */
	MaxPool maxPool2(2, 2, 2, 2);
	/* B x 64 x 7 x 7 */
	Flatten flatten;
	/* B x 64*7*7 x 1 x 1 */
	Conv conv3(10, 16*7*7, 1, 1);
	/* B x 10 x 1 x 1*/
	FlattenRank flattenRank;
	/* B x 10 */
	SoftMax softMax;
	/* B x 10 */

	//auto opt1 = std::make_shared<SgdWithMomentum>(learning_rate, 0.9, std::tuple<int, int, int, int>{4, 1, 3, 3});
	//auto opt2 = std::make_shared<SgdWithMomentum>(learning_rate, 0.9, std::tuple<int, int, int, int>{8, 4, 3, 3});
	//auto opt3 = std::make_shared<SgdWithMomentum>(learning_rate, 0.9, std::tuple<int, int, int, int>{10, 8*7*7, 1, 1});

	auto opt1 = std::make_shared<Sgd>(learning_rate);
	auto opt2 = std::make_shared<Sgd>(learning_rate);
	auto opt3 = std::make_shared<Sgd>(learning_rate);

	conv1.setOptimizer(opt1);
	conv2.setOptimizer(opt2);
	conv3.setOptimizer(opt3);

	auto initializer = std::make_shared<UniformRandom>();
	conv1.setInitializer(initializer);
	conv2.setInitializer(initializer);
	conv3.setInitializer(initializer);

	CrossEntropyLoss lossLayer;

	auto train = [&](int i) -> float {
		auto batch = loader.getBatch(batchSize);
		auto labels = batch.first;
		auto images = batch.second;
		assertSize(images, batchSize, 1, 28, 28);

		// Forward:
		auto tensor1 = reLu1.forward(conv1.forward(images));
		// assertSize(tensor1, batchSize, 32, 28, 28);
		tensor1 = maxPool1.forward(tensor1);
		// assertSize(tensor1, batchSize, 32, 14, 14);

		auto tensor2 = reLu2.forward(conv2.forward(tensor1));
		// assertSize(tensor2, batchSize, 64, 14, 14);
		tensor2 = maxPool2.forward(tensor2);
		// assertSize(tensor2, batchSize, 64, 7, 7);

		auto tensor3 = conv3.forward(flatten.forward(tensor2));
		// assertSize(tensor3, batchSize, 10, 1, 1);
		auto res = softMax.forward(flattenRank.forward(tensor3));
		
		// assertSize(res, batchSize, 10);

		float loss = lossLayer.forward(res, labels);
		if (i % 25 == 0) {
			float correct = correctDigits(labels, res);
			printf("Loss: %f, correct: %f\n", loss, correct);
		}

		// Backward:

		auto loss_err = lossLayer.backward(labels);
		auto err = flattenRank.backward(softMax.backward(loss_err));
		// assertSize(err, batchSize, 10, 1, 1);

		auto err3 = conv3.backward(err);
		// assertSize(err3, batchSize, 147, 1, 1);

		auto err2 = maxPool2.backward(flatten.backward(err3));
		// assertSize(err2, batchSize, 3, 14, 14);

		err2 = conv2.backward(reLu2.backward(err2));
		// assertSize(err2, batchSize, 3, 14, 14);

		auto err1 = maxPool1.backward(err2);
		// assertSize(err1, batchSize, 3, 28, 28);

		err1 = conv1.backward(reLu1.backward(err1));
		// assertSize(err1, batchSize, 1, 28, 28);

		return loss;
	};

	for (int i = 0; i < iterations; i++)
		train(i);

	return 0; 
}




