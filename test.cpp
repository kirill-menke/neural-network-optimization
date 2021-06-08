#include "HPC/Helper.h"
#include "HPC/ReLU.h"
#include "HPC/MaxPool.h"
#include "HPC/SoftMax.h"
#include "HPC/Loss.h"

#include <stdio.h>
#include <stdlib.h>

int main() {
	printf("Hello World!\n");

	CrossEntropyLoss lossLayer;

	auto testInput = std::make_shared<Eigen::Tensor<float, 2>>(5, 5);
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			(*testInput)(i, j) = 0.01;
		}
		(*testInput)(i, 4 - i) = 0.96;
	}

	auto testLabels = std::make_shared<Eigen::Tensor<float, 2>>(5, 5);
	testLabels->setZero();
	for (int i = 0; i < 5; i++) {
		(*testLabels)(i, i) = 1.;
	}

	auto loss = lossLayer.forward(testInput, testLabels);
	printf("Loss: %f\n", loss);

	auto error_tensor = lossLayer.backward(testLabels);
	printTensor(*error_tensor);

#if 0
	MNISTLoader mnist("./data/mnist-train.txt");

	int batchSize = 3;
	auto batch = mnist.loadBatch(batchSize);
	auto numbers = batch.first;

	for (int i = 0; i < batchSize; i++)
		printf("Number #%d: %d\n", i, numbers[i]);

	printTensor(*batch.second);
#endif

#if 0
	auto testInput = std::make_shared<Eigen::Tensor<float, 4>>(1, 1, 12, 12);
	for (int x = 0; x < 12; x++) {
		for (int y = 0; y < 12; y++) {
			(*testInput)(0, 0, x, y) = rand() % 100 - 50;
		}
	}

	printTensor(*testInput);

	// MaxPool maxPool(12, 12, 4, 4, 3, 3);
	// auto testOutput = maxPool.forward(testInput);
	ReLU reLu;
	auto testOutput = reLu.forward(testInput);

	printTensor(*testOutput);

	// auto testError = std::make_shared<Eigen::Tensor<float, 4>>(1, 1, 3, 3);
	// for (int x = 0; x < 3; x++) {
	//	for (int y = 0; y < 3; y++) {
	//		(*testError)(0, 0, x, y) = rand() % 100 - 50;
	//	}
	// }

	auto testError = std::make_shared<Eigen::Tensor<float, 4>>(1, 1, 12, 12);
	for (int x = 0; x < 12; x++) {
		for (int y = 0; y < 12; y++) {
			(*testError)(0, 0, x, y) = rand() % 100 - 50;
		}
	}

	printTensor(*testError);

	// auto nextError = maxPool.backward(testError);
	auto nextError = reLu.backward(testError);

	printTensor(*nextError);
#endif

#if 0
	auto testInput = std::make_shared<Eigen::Tensor<float, 2>>(5, 10);
	for (int b = 0; b < 5; b++) {
		for (int f = 0; f < 10; f++) {
			(*testInput)(b, f) = 0.;
		}
		(*testInput)(b, b) = 1.;
	}

	printTensor(*testInput);

	SoftMax softMax;
	auto testOutput = softMax.forward(testInput);

	printTensor(*testOutput);

	auto testError = std::make_shared<Eigen::Tensor<float, 2>>(5, 10);
	for (int b = 0; b < 5; b++) {
		for (int f = 0; f < 10; f++) {
			(*testError)(b, f) = 0.;
		}
		(*testError)(b, b + 1) = 1.;
	}

	printTensor(*testError);

	auto nextError = softMax.backward(testError);

	printTensor(*nextError);
#endif

	return EXIT_SUCCESS;
}

