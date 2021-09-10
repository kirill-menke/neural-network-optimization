#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>

#include "optimizer.h"
#include "initializer.h"
#include "maxpool.h"
#include "flatten.h"
#include "relu.h"
#include "softmax.h"
#include "loss.h"
#include "conv.h"
#include "dropout.h"
#include "utils.h"

int main(int argc, const char* argv[]) {

	if (argc != 2) {
		fprintf(stderr, "usage: %s <iterations>\n", argv[0]);
		return EXIT_FAILURE;
	}

	/* CNN parameters */
	const int iterations = atoi(argv[1]);
	const int batch_size = 8;
	const float learning_rate = 0.001f;

	/* CNN layers */
	Conv conv1(8, 1, 3, 1);
	ReLU reLu1;
	MaxPool maxPool1(2, 2, 2, 2);
	Dropout drop1(0.8);

	Conv conv2(16, 8, 3, 1);
	ReLU reLu2;
	MaxPool maxPool2(2, 2, 2, 2);
	Dropout drop2(0.8);

	Flatten flatten;
	Conv conv3(10, 16 * 7 * 7, 1, 1);
	FlattenRank flatten_rank;

	SoftMax soft_max;
	CrossEntropyLoss loss_layer;

	/* Set optimizer for all trainable layers */
	Adam opt1(learning_rate, 0.9, 0.999, conv1.getWeightDims());
	Adam opt2(learning_rate, 0.9, 0.999, conv2.getWeightDims());
	Adam opt3(learning_rate, 0.9, 0.999, conv3.getWeightDims());
	conv1.setOptimizer(&opt1);
	conv2.setOptimizer(&opt2);
	conv3.setOptimizer(&opt3);

	/* Set initializer for all trainable layers */
	He he_init;
	conv1.setInitializer(&he_init);
	conv2.setInitializer(&he_init);
	conv3.setInitializer(&he_init);

	/* Load data and split into train and test set */
	MNISTLoader loader("../data/mnist/mnist-train.txt");
	loader.loadFullDataset(0.05, true);


	/* Run network on test data an check accuracy */
	auto test = [&]() {
		auto test_data = loader.getTestSet();
		auto test_labels = test_data.first;
		auto test_images = test_data.second;

		drop1.setTestingPhase(true);
		drop2.setTestingPhase(true);

		auto tensor1 = reLu1.forward(conv1.forward(test_images));
		tensor1 = drop1.forward(maxPool1.forward(tensor1));

		auto tensor2 = reLu2.forward(conv2.forward(tensor1));
		tensor2 = drop2.forward(maxPool2.forward(tensor2));

		auto tensor3 = conv3.forward(flatten.forward(tensor2));
		auto res = soft_max.forward(flatten_rank.forward(tensor3));

		float accuracy = calculateAccuracy(test_labels, res);

		std::cout << "Accuracy: " << accuracy << std::endl;
	};


	/* Train network */
	auto train = [&](int i) -> float {
		auto batch = loader.getBatch(batch_size);
		auto labels = batch.first;
		auto images = batch.second;
		assertSize(images, batch_size, 1, 28, 28);
		assertSize(labels, batch_size, 10);


		/***			Forward				***/
		auto tensor1 = reLu1.forward(conv1.forward(images));
		tensor1 = drop1.forward(maxPool1.forward(tensor1));

		auto tensor2 = reLu2.forward(conv2.forward(tensor1));
		tensor2 = drop2.forward(maxPool2.forward(tensor2));

		auto tensor3 = conv3.forward(flatten.forward(tensor2));
		auto res = soft_max.forward(flatten_rank.forward(tensor3));

		float loss = loss_layer.forward(res, labels);


		/***			Backward			***/
		auto loss_err = loss_layer.backward(labels);

		auto err = flatten_rank.backward(soft_max.backward(loss_err));
		auto err3 = conv3.backward(err);

		auto err2 = maxPool2.backward(drop2.backward(flatten.backward(err3)));
		err2 = conv2.backward(reLu2.backward(err2));
		
		auto err1 = maxPool1.backward(drop1.backward(err2));
		err1 = conv1.backward(reLu1.backward(err1));


		/***		Print Measurements			***/
		if (i % 25 == 0) {
			float accuracy = calculateAccuracy(labels, res);
			printf("Loss: %f, correct: %f\n", loss, accuracy);
		}

		if (i % 100 == 0) {
			test();
		}

		return loss;
	};


	for (int i = 0; i < iterations; i++)
		train(i);


	return 0;
}




