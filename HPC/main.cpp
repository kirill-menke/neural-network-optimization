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

int main() {
	int iterations = 200;
	int batchSize = 10;
	float learning_rate = 1e-4f;

	MNISTLoader loader("./data/mnist-train.txt");

	/* B x 1 x 28 x 28 */
	Conv conv1(6, 1, 3, 1);
	/* B x 6 x 28 x 28 */
	ReLU reLu1;
	/* B x 6 x 28 x 28 */
	MaxPool maxPool1(2, 2, 2, 2);
	/* B x 6 x 14 x 14 */
	Conv conv2(12, 6, 3, 1);
	/* B x 12 x 14 x 14 */
	ReLU reLu2;
	/* B x 12 x 14 x 14 */
	MaxPool maxPool2(2, 2, 2, 2);
	/* B x 12 x 7 x 7 */
	Flatten flatten;
	/* B x 588 x 1 x 1 */
	Conv conv3(10, 588, 1, 1);
	/* B x 10 x 1 x 1*/
	FlattenRank flattenRank;
	/* B x 10 */
	SoftMax softMax;
	/* B x 10 */

	auto opt1 = std::make_shared<Sgd>(learning_rate);
	conv1.setOptimizer(opt1);
	auto opt2 = std::make_shared<Sgd>(learning_rate);
	conv2.setOptimizer(opt2);
	auto opt3 = std::make_shared<Sgd>(learning_rate);
	conv3.setOptimizer(opt3);

	CrossEntropyLoss lossLayer;

	auto train = [&](){
		auto batch = loader.loadBatch(batchSize);
		auto labels = batch.first;
		auto images = batch.second;
		assertSize(images, batchSize, 1, 28, 28);

		// Forward:

		auto tensor1 = reLu1.forward(conv1.forward(images));
		assertSize(tensor1, batchSize, 6, 28, 28);
		tensor1 = maxPool1.forward(tensor1);
		assertSize(tensor1, batchSize, 6, 14, 14);

		auto tensor2 = reLu2.forward(conv2.forward(tensor1));
		assertSize(tensor2, batchSize, 12, 14, 14);
		tensor2 = maxPool2.forward(tensor2);
		assertSize(tensor2, batchSize, 12, 7, 7);

		auto tensor3 = conv3.forward(flatten.forward(tensor2));
		assertSize(tensor3, batchSize, 10, 1, 1);
		auto res = softMax.forward(flattenRank.forward(tensor3));
		assertSize(res, batchSize, 10);

		float loss = lossLayer.forward(res, labels);
		printf("Loss: %f\n", loss);

		// Backward:

		auto err = flattenRank.backward(softMax.backward(lossLayer.backward(labels)));
		assertSize(err, batchSize, 10, 1, 1);
		auto err3 = conv3.backward(err);
		assertSize(err3, batchSize, 588, 1, 1);

		auto err2 = maxPool2.backward(flatten.backward(err3));
		assertSize(err2, batchSize, 12, 14, 14);
		err2 = conv2.backward(reLu2.backward(err2));
		assertSize(err2, batchSize, 6, 14, 14);

		auto err1 = maxPool1.backward(err2);
		assertSize(err1, batchSize, 6, 28, 28);
		err1 = conv1.backward(reLu1.backward(err1));
		assertSize(err1, batchSize, 1, 28, 28);
	};

	for (int i = 0; i < iterations; i++)
		train();

}




