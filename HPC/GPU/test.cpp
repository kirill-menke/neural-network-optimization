#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <unsupported/Eigen/CXX11/Tensor>

#include "./GPUConv.h"
#include "./GPUUpsample.h"
#include "./GPUTransConv.h"
#include "./GPUReLU.h"
#include "./GPUMaxPool.h"
#include "./GPUOptimizer.h"
#include "./GPULoss.h"
#include "./GPUSoftMax.h"
#include "./utils.h"

#include "../SoftMax.h"
#include "../Loss.h"
#include "../Helper.h"

#include "./Tensor.h"



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

float sum(Tensor<float, 2> &t) {
	t.moveToHost();
	double sum = 0.;
	for (int x = 0; x < t.dim(0); x++)
		for (int y = 0; y < t.dim(1); y++)
			sum += t(x, y);

	return sum;
}

std::pair<Tensor<float, 4>, Tensor<float, 3>> testData(int size) {
	Tensor<float, 4> image({ 1, 1, size, size });
	Tensor<float, 3> labels({ 1, size, size });

	for (int x = 0; x < size; x++) {
		for (int y = 0; y < size; y++) {
			image(0, 0, x, y) = 0.;
			labels(0, x, y) = 1.;
		}
	}

	image.moveToDevice();
	labels.moveToDevice();
	return { image, labels };
}

int main() {
#if 1
	int batchSize = 1;
	int iterations = 100;
	float learning_rate = 0.005f;

	GPUConv conv1(1, 2, 100, 100, 3, 3, 1, 1);
	GPUReLU reLu1;

	GPUConv conv2(2, 4, 100, 100, 3, 3, 1, 1);
	GPUReLU reLu2;

	GPUConv conv3(4, 2, 100, 100, 3, 3, 1, 1);
	GPUSoftMax softMax;

	GPUCrossEntropyLoss loss(100, 100);

	MembraneLoader dataLoader("../../data/cell-membranes/");
	conv1.optimizer = new GPUSgd(learning_rate);
	conv2.optimizer = new GPUSgd(learning_rate);
	conv3.optimizer = new GPUSgd(learning_rate);
	uniformRandomInit(-0.5, 0.5, conv1.weights, conv1.bias);
	uniformRandomInit(-0.5, 0.5, conv2.weights, conv2.bias);
	uniformRandomInit(-0.5, 0.5, conv3.weights, conv3.bias);

	auto train = [&](int i) -> float {
		// printf("--> Iteration #%d\n", i);

		// auto data = dataLoader.loadBatch(batchSize);
		auto data = testData(100);

		auto t1 = reLu1.forward(conv1.forward(data.first));

		// t1.moveToHost();
		// t1.dump4D(stdout, "t1");

		auto t2 = reLu2.forward(conv2.forward(t1));
		
		// t2.moveToHost();
		// t2.dump4D(stdout, "t2");

		auto pred = softMax.forward(conv3.forward(t2));

		// pred.moveToHost();
		// pred.dump4D(stdout, "pred");

		auto l = loss.forward(pred, data.second);

		// l.moveToHost();
		// l.dump2D(stdout, "loss");

		printf("#%d:\tloss=%f\n", i, sum(l));

		auto err = loss.backward(pred, data.second);

		// err.moveToHost();
		// err.dump4D(stdout, "loss.backward");

		auto err3 = conv3.backward(softMax.backward(err));

		// err3.moveToHost();
		// err3.dump4D(stdout, "conv3.backward");

		auto err2 = conv2.backward(reLu2.backward(err3));

		// err2.moveToHost();
		// err2.dump4D(stdout, "conv2.backward");

		auto err1 = conv1.backward(reLu1.backward(err2));

		// err1.moveToHost();
		// err1.dump4D(stdout, "conv1.backward");

		return 0.;
	};

	for (int i = 0; i < iterations; i++)
		train(i);
#endif
#if 0
	int batchSize = 5;
	int iterations = 10;
	float learning_rate = 0.0005f;

	GPUConv conv1(1, 8, 512, 512, 3, 3, 1, 1);
	GPUReLU reLu1;
	GPUMaxPool pool1(batchSize, 8, 512, 512, 2, 2);

	GPUConv conv2(8, 16, 256, 256, 3, 3, 1, 1);
	GPUReLU reLu2;
	GPUMaxPool pool2(batchSize, 16, 256, 256, 2, 2);

	GPUConv conv3(16, 32, 128, 128, 3, 3, 1, 1);
	GPUReLU reLu3;
	GPUMaxPool pool3(batchSize, 32, 128, 128, 2, 2);

	GPUConv conv4(32, 64, 64, 64, 3, 3, 1, 1);
	GPUReLU reLu4;
	GPUUpsample up4(64, 64, 2, 2);

	// Join results of up4 and reLu3!
	GPUConv conv5(64 + 32, 64, 128, 128, 3, 3, 1, 1);
	GPUReLU reLu5;
	GPUUpsample up5(128, 128, 2, 2);

	// Join results of up5 and reLu2!
	GPUConv conv6(64 + 16, 32, 256, 256, 3, 3, 1, 1);
	GPUReLU reLu6;
	GPUUpsample up6(256, 256, 2, 2);

	// Join results of up6 and reLu1!
	GPUConv conv7(32 + 8, 16, 512, 512, 3, 3, 1, 1);
	GPUReLU reLu7;

	GPUConv conv8(16, 2, 512, 512, 3, 3, 1, 1);
	GPUSoftMax softMax;

	GPUCrossEntropyLoss loss(512, 512);

	MembraneLoader dataLoader("../../data/cell-membranes/");
	conv1.optimizer = new GPUSgd(learning_rate);
	conv2.optimizer = new GPUSgd(learning_rate);
	conv3.optimizer = new GPUSgd(learning_rate);
	conv4.optimizer = new GPUSgd(learning_rate);
	conv5.optimizer = new GPUSgd(learning_rate);
	conv6.optimizer = new GPUSgd(learning_rate);
	conv7.optimizer = new GPUSgd(learning_rate);
	conv8.optimizer = new GPUSgd(learning_rate);
	uniformRandomInit(-0.025, 0.025, conv1.weights, conv1.bias);
	uniformRandomInit(-0.025, 0.025, conv2.weights, conv2.bias);
	uniformRandomInit(-0.025, 0.025, conv3.weights, conv3.bias);
	uniformRandomInit(-0.025, 0.025, conv4.weights, conv4.bias);
	uniformRandomInit(-0.025, 0.025, conv5.weights, conv5.bias);
	uniformRandomInit(-0.025, 0.025, conv6.weights, conv6.bias);
	uniformRandomInit(-0.025, 0.025, conv7.weights, conv7.bias);
	uniformRandomInit(-0.025, 0.025, conv8.weights, conv8.bias);

	auto train = [&](int i) -> float {
		// printf("--> Iteration #%d\n", i);

		auto data = dataLoader.loadBatch(batchSize);
		auto t0 = data.first;

		auto t1 = reLu1.forward(conv1.forward(t0));
		auto t1pooled = pool1.forward(t1);

		// printf("f1: %d x %d x %d\n", t1.dim(1), t1.dim(2), t1.dim(3));

		auto t2 = reLu2.forward(conv2.forward(t1pooled));
		auto t2pooled = pool2.forward(t2);

		// printf("f2: %d x %d x %d\n", t2.dim(1), t2.dim(2), t2.dim(3));

		auto t3 = reLu3.forward(conv3.forward(t2pooled));
		auto t3pooled = pool3.forward(t3);

		// printf("f3: %d x %d x %d\n", t3.dim(1), t3.dim(2), t3.dim(3));

		auto t4 = reLu4.forward(conv4.forward(t3pooled));
		auto t4upped = up4.forward(t4);

		// printf("f4: %d x %d x %d\n", t4.dim(1), t4.dim(2), t4.dim(3));

		auto t5in = mergeAtChannelDim(t4upped, t3);
		auto t5 = reLu5.forward(conv5.forward(t5in));
		auto t5upped = up5.forward(t5);

		// printf("f5: %d x %d x %d\n", t5.dim(1), t5.dim(2), t5.dim(3));

		auto t6in = mergeAtChannelDim(t5upped, t2);
		auto t6 = reLu6.forward(conv6.forward(t6in));
		auto t6upped = up6.forward(t6);

		// printf("f6: %d x %d x %d\n", t6.dim(1), t6.dim(2), t6.dim(3));

		auto t7in = mergeAtChannelDim(t6upped, t1);
		auto t7 = reLu7.forward(conv7.forward(t7in));

		// printf("f7: %d x %d x %d\n", t7.dim(1), t7.dim(2), t7.dim(3));

		auto t8 = conv8.forward(t7);
		auto pred = softMax.forward(t8);

		// pred.moveToHost();
		// pred.dump4D(fopen("./pred.tensor.txt", "w"));

		auto l = loss.forward(pred, data.second);
		printf("#%d:\tloss=%f\n", i, sum(l));

		// l.moveToHost();
		// l.dump2D(fopen("./loss.tensor.txt", "w"));
		auto err = loss.backward(data.second);
		// err.moveToHost();
		// err.dump4D(fopen("./loss-backward.tensor.txt", "w"));
		auto err9 = softMax.backward(err);
		// err9.moveToHost();
		// err9.dump4D(fopen("./softmax-backward.tensor.txt", "w"));
		auto err8 = conv8.backward(err9);
		// err8.moveToHost();
		// err8.dump4D(fopen("./conv8-backward.tensor.txt", "w"));

		// printf("b1: %d x %d x %d\n", err8.dim(1), err8.dim(2), err8.dim(3));

		auto err7 = conv7.backward(reLu7.backward(err8));
		auto err7split = splitAtChannelDim(err7, 32);

		// printf("b2: %d x %d x %d\n", err7.dim(1), err7.dim(2), err7.dim(3));

		auto err6downed = up6.backward(err7split.first);
		auto err6 = conv6.backward(reLu6.backward(err6downed));
		auto err6split = splitAtChannelDim(err6, 64);

		// printf("b3: %d x %d x %d\n", err6.dim(1), err6.dim(2), err6.dim(3));

		auto err5downed = up5.backward(err6split.first);
		auto err5 = conv5.backward(reLu5.backward(err5downed));
		auto err5split = splitAtChannelDim(err5, 64);

		// printf("b4: %d x %d x %d\n", err5.dim(1), err5.dim(2), err5.dim(3));

		auto err4downed = up4.backward(err5split.first);
		auto err4 = conv4.backward(reLu4.backward(err4downed));

		// printf("b5: %d x %d x %d\n", err4.dim(1), err4.dim(2), err4.dim(3));

		auto err3upped = pool3.backward(err4);
		auto err3in = err3upped + err5split.second;
		auto err3 = conv3.backward(reLu3.backward(err3in));

		// printf("b6: %d x %d x %d\n", err3.dim(1), err3.dim(2), err3.dim(3));

		auto err2upped = pool2.backward(err3);
		auto err2in = err2upped + err6split.second;
		auto err2 = conv2.backward(reLu2.backward(err2in));

		// printf("b7: %d x %d x %d\n", err2.dim(1), err2.dim(2), err2.dim(3));

		auto err1upped = pool1.backward(err2);
		auto err1in = err1upped + err7split.second;
		auto err1 = conv1.backward(reLu1.backward(err1in));

		// printf("b8: %d x %d x %d\n", err1.dim(1), err1.dim(2), err1.dim(3));

		// err1.moveToHost();
		// err1.dump4D(stdout, "Error");

		return 0.;
	};

	for (int i = 0; i < iterations; i++)
		train(i);
#endif
#if 0
	int iterations = 1000;
	int batchSize = 10;
	float learning_rate = 0.01f;

	MNISTLoader mnist("../data/mnist-train.txt");

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
	return EXIT_SUCCESS;
}

