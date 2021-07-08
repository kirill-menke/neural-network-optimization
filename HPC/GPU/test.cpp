#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <unsupported/Eigen/CXX11/Tensor>

#include "./GPUConv.h"
#include "./GPUReLU.h"
#include "./GPUMaxPool.h"
#include "./GPUOptimizer.h"
#include "./utils.h"

#include "../SoftMax.h"
#include "../Loss.h"
#include "../Helper.h"


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

int main() {
#if 1
	int iterations = 1000;
	int batchSize = 10;
	float learning_rate = 0.01;

	MNISTLoader mnist("../../data/mnist-train.txt");

	printf("Data loaded...\n");

	GPUConv conv1(1, 8, 28, 28, 3, 3, 1, 1);
	GPUMaxPool maxPool1(batchSize, 8, 28, 28, 2, 2);
	GPUReLU reLu1;

	GPUConv conv2(8, 16, 14, 14, 3, 3, 1, 1);
	GPUMaxPool maxPool2(batchSize, 16, 14, 14, 2, 2);
	GPUReLU reLu2;

	GPUConv conv3(16*7*7, 10, 1, 1, 1, 1, 1, 1);

	FlattenRank flattenRank;
	SoftMax softMax;
	CrossEntropyLoss loss;

	printf("Initializing weights...\n");

	conv1.optimizer = new GPUSgd(learning_rate);
	uniformRandomInit(-0.05, 0.05, conv1.weights, conv2.bias);

	conv2.optimizer = new GPUSgd(learning_rate);
	uniformRandomInit(-0.05, 0.05, conv2.weights, conv2.bias);

	conv3.optimizer = new GPUSgd(learning_rate);
	uniformRandomInit(-0.05, 0.05, conv3.weights, conv3.bias);

	auto train = [&](int i) -> float {
		// printf("Iteration #%d\n");
		auto batch = mnist.loadBatch(batchSize);
		auto labels = batch.first;
		auto images = fromEigenTensor(*batch.second);

		auto t1 = conv1.forward(images);
		auto t2 = maxPool1.forward(t1);
		auto t3 = reLu1.forward(t2);
		auto t4 = conv2.forward(t3);
		auto t5 = maxPool2.forward(t4);
		auto t6 = reLu2.forward(t5);
		auto t7 = t6.reshape({ batchSize, 16*7*7, 1, 1 });
		auto t8 = conv3.forward(t7);
		auto t9 = flattenRank.forward(toEigenTensor(t8));
		auto res = softMax.forward(t9);

		auto l = loss.forward(res, labels);

		printf("#%d:\tloss=%f\n", i, l);

		auto err0 = loss.backward(labels);
		auto err1 = flattenRank.backward(softMax.backward(err0));
		auto err2 = fromEigenTensor(*err1);
		auto err3 = conv3.backward(err2);
		auto err4 = err3.reshape({ batchSize, 16, 7, 7 });
		auto err5 = reLu2.backward(err4);
		auto err6 = maxPool2.backward(err5);
		auto err7 = conv2.backward(err6);
		auto err8 = reLu1.backward(err7);
		auto err9 = maxPool1.backward(err8);
		auto err  = conv1.backward(err9);

		return l;
	};

	for (int i = 0; i < iterations; i++)
		train(i);
#endif
#if 0
	srand(42);

	GPUReLU reLu;
	GPUMaxPool maxPool(1, 1, 6, 6, 2, 2);

	Tensor<float, 4> t1({ 1, 1, 6, 6 });
	for (int x = 0; x < 6; x++)
		for (int y = 0; y < 6; y++)
			t1(0, 0, x, y) = rand() % 10 - 5;

	//Tensor<float, 4> t2({ 1, 1, 3, 3 });
	//for (int x = 0; x < 3; x++)
	//	for (int y = 0; y < 3; y++)
	//		t2(0, 0, x, y) = rand() % 10;

	Tensor<float, 4> t2({ 1, 1, 6, 6 });
	for (int x = 0; x < 6; x++)
		for (int y = 0; y < 6; y++)
			t2(0, 0, x, y) = rand() % 10 - 5;


	t1.dump4D(stdout, "t1");
	t2.dump4D(stdout, "t2");
	t1.moveToDevice();
	t2.moveToDevice();

	//auto maxPoolRes1 = maxPool.forward(t1);
	//maxPoolRes1.moveToHost();
	//maxPoolRes1.dump4D(stdout, "maxPoolRes (forward)");

	//auto maxPoolRes2 = maxPool.backward(t2);
	//maxPoolRes2.moveToHost();
	//maxPoolRes2.dump4D(stdout, "maxPoolRes (backward)");

	auto reLuRes1 = reLu.forward(t1);
	reLuRes1.moveToHost();
	reLuRes1.dump4D(stdout, "reLuRes (forward)");

	auto reLuRes2 = reLu.backward(t2);
	reLuRes2.moveToHost();
	reLuRes2.dump4D(stdout, "reLuRes (backward)");
#endif
	return EXIT_SUCCESS;
}

