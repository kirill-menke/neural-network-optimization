#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <random>

#include <unsupported/Eigen/CXX11/Tensor>

#include "./GPUConv.h"
#include "./GPUReLU.h"
#include "./GPUMaxPool.h"
#include "./GPUOptimizer.h"

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

std::shared_ptr<Eigen::Tensor<float, 4>> toEigenTensor(Tensor<float, 4> tensor) {
	tensor.moveToHost();

	auto eigenTensor = std::make_shared<Eigen::Tensor<float, 4>>(
			tensor.dim(0), tensor.dim(1), tensor.dim(2), tensor.dim(3));
	for (int b = 0; b < tensor.dim(0); b++) {
		for (int c = 0; c < tensor.dim(1); c++) {
			for (int x = 0; x < tensor.dim(2); x++) {
				for (int y = 0; y < tensor.dim(3); y++) {
					(*eigenTensor)(b, c, x, y) = tensor(b, c, x, y);
				}
			}
		}
	}

	return eigenTensor;
}

Tensor<float, 4> fromEigenTensor(Eigen::Tensor<float, 4> &eigenTensor) {
	Tensor<float, 4> tensor(Tensor<float, 4>::ON_CPU, {
		int(eigenTensor.dimension(0)),
		int(eigenTensor.dimension(1)),
		int(eigenTensor.dimension(2)),
		int(eigenTensor.dimension(3)) });

	for (int b = 0; b < tensor.dim(0); b++) {
		for (int c = 0; c < tensor.dim(1); c++) {
			for (int x = 0; x < tensor.dim(2); x++) {
				for (int y = 0; y < tensor.dim(3); y++) {
					tensor(b, c, x, y) = eigenTensor(b, c, x, y);
				}
			}
		}
	}

	tensor.moveToDevice();
	return tensor;
}

int main() {
	int iterations = 1;
	int batchSize = 1;
	float learning_rate = 0.005;

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
	
	std::mt19937_64 rng(0);
	std::uniform_real_distribution<float> unif(-0.01, 0.01);

	conv1.optimizer = new GPUSgd(learning_rate);
	conv1.weights.allocCPU();
	conv1.bias.allocCPU();
	for (int f = 0; f < 1; f++) {
		for (int c = 0; c < 8; c++)
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
					conv1.weights(f, c, x, y) = unif(rng);
		conv1.bias(f) = 0.;
	}
	conv1.weights.moveToDevice();
	conv1.bias.moveToDevice();

	conv2.optimizer = new GPUSgd(learning_rate);
	conv2.weights.allocCPU();
	conv2.bias.allocCPU();
	for (int f = 0; f < 8; f++) {
		for (int c = 0; c < 16; c++)
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
					conv2.weights(f, c, x, y) = unif(rng);
		conv2.bias(f) = 0.;
	}
	conv2.weights.moveToDevice();
	conv2.bias.moveToDevice();

	conv3.optimizer = new GPUSgd(learning_rate);
	conv3.weights.allocCPU();
	conv3.bias.allocCPU();
	for (int f = 0; f < 10; f++) {
		for (int c = 0; c < 16*7*7; c++)
			conv3.weights(f, c, 0, 0) = unif(rng);
		conv3.bias(f) = 0.;
	}
	conv3.weights.moveToDevice();
	conv3.bias.moveToDevice();

	auto train = [&](int i) -> float {
		printf("Iteration #%d\n");
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

		printf("Loss: %f\n", l);

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
		auto _    = conv1.backward(err9);

		return l;
	};

	for (int i = 0; i < iterations; i++)
		train(i);

#if 0	
	int batchSize = 1;
	int inputChannels = 1;
	int outputChannels = 1;
	int filterWidth = 3;
	int filterHeight = 3;
	int imageWidth = 5;
	int imageHeight = 5;
	int strideX = 1;
	int strideY = 1;

	GPUConv convLayer(
		inputChannels,
		imageWidth,
		imageHeight,
		outputChannels,
		filterWidth,
		filterHeight,
		strideX,
		strideY);

	convLayer.filters.moveToHost();
	convLayer.filters(0, 0, 0, 0) = 0.;
	convLayer.filters(0, 0, 1, 0) = 1.;
	convLayer.filters(0, 0, 2, 0) = 2.;
	convLayer.filters(0, 0, 0, 1) = 0.;
	convLayer.filters(0, 0, 1, 1) = 1.;
	convLayer.filters(0, 0, 2, 1) = 2.;
	convLayer.filters(0, 0, 0, 2) = 0.;
	convLayer.filters(0, 0, 1, 2) = 1.;
	convLayer.filters(0, 0, 2, 2) = 2.;
	convLayer.filters.dump(stdout, "Filter");
	convLayer.filters.moveToDevice();

	Tensor<float, 4> imageIn(Tensor<float, 4>::ON_CPU, { batchSize, inputChannels, imageWidth, imageHeight });
	imageIn(0, 0, 0, 0) = 0.;
	imageIn(0, 0, 1, 0) = 1.;
	imageIn(0, 0, 2, 0) = 2.;
	imageIn(0, 0, 3, 0) = 3.;
	imageIn(0, 0, 4, 0) = 4.;
	imageIn(0, 0, 0, 1) = 5.;
	imageIn(0, 0, 1, 1) = 6.;
	imageIn(0, 0, 2, 1) = 7.;
	imageIn(0, 0, 3, 1) = 8.;
	imageIn(0, 0, 4, 1) = 9.;
	imageIn(0, 0, 0, 2) = 10.;
	imageIn(0, 0, 1, 2) = 11.;
	imageIn(0, 0, 2, 2) = 12.;
	imageIn(0, 0, 3, 2) = 13.;
	imageIn(0, 0, 4, 2) = 14.;
	imageIn(0, 0, 0, 3) = 15.;
	imageIn(0, 0, 1, 3) = 16.;
	imageIn(0, 0, 2, 3) = 17.;
	imageIn(0, 0, 3, 3) = 18.;
	imageIn(0, 0, 4, 3) = 19.;
	imageIn(0, 0, 0, 4) = 20.;
	imageIn(0, 0, 1, 4) = 21.;
	imageIn(0, 0, 2, 4) = 22.;
	imageIn(0, 0, 3, 4) = 23.;
	imageIn(0, 0, 4, 4) = 24.;
	imageIn.dump(stdout, "Input Image");
	imageIn.moveToDevice();

	Tensor<float, 4> imageOut = convLayer.forward(imageIn);
	imageOut.moveToHost();
	imageOut.dump(stdout, "Output Image");

	Tensor<float, 4> error_tensor(Tensor<float, 4>::ON_CPU, { batchSize, outputChannels, imageWidth, imageHeight });
	error_tensor(0, 0, 0, 0) = 0.;
	error_tensor(0, 0, 1, 0) = 1.;
	error_tensor(0, 0, 2, 0) = 2.;
	error_tensor(0, 0, 3, 0) = 3.;
	error_tensor(0, 0, 4, 0) = 4.;
	error_tensor(0, 0, 0, 1) = 5.;
	error_tensor(0, 0, 1, 1) = 6.;
	error_tensor(0, 0, 2, 1) = 7.;
	error_tensor(0, 0, 3, 1) = 8.;
	error_tensor(0, 0, 4, 1) = 9.;
	error_tensor(0, 0, 0, 2) = 10.;
	error_tensor(0, 0, 1, 2) = 11.;
	error_tensor(0, 0, 2, 2) = 12.;
	error_tensor(0, 0, 3, 2) = 13.;
	error_tensor(0, 0, 4, 2) = 14.;
	error_tensor(0, 0, 0, 3) = 15.;
	error_tensor(0, 0, 1, 3) = 16.;
	error_tensor(0, 0, 2, 3) = 17.;
	error_tensor(0, 0, 3, 3) = 18.;
	error_tensor(0, 0, 4, 3) = 19.;
	error_tensor(0, 0, 0, 4) = 20.;
	error_tensor(0, 0, 1, 4) = 21.;
	error_tensor(0, 0, 2, 4) = 22.;
	error_tensor(0, 0, 3, 4) = 23.;
	error_tensor(0, 0, 4, 4) = 24.;
	error_tensor.dump(stdout, "Error Tensor");
	error_tensor.moveToDevice();

	Tensor<float, 4> next_error_tensor = convLayer.backward(error_tensor);
	next_error_tensor.moveToHost();
	next_error_tensor.dump(stdout, "Next Error Tensor");
#endif

#if 0
	int batchSize = 1;
	int inputChannels = 1;
	int outputChannels = 1;
	int filterWidth = 3;
	int filterHeight = 3;
	int imageWidth = 5;
	int imageHeight = 5;
	int strideX = 2;
	int strideY = 2;

	GPUConv convLayer(
		inputChannels,
		imageWidth,
		imageHeight,
		outputChannels,
		filterWidth,
		filterHeight,
		strideX,
		strideY);

	convLayer.weights.moveToHost();
	convLayer.weights(0, 0, 0, 0) = 0.;
	convLayer.weights(0, 0, 1, 0) = 1.;
	convLayer.weights(0, 0, 2, 0) = 2.;
	convLayer.weights(0, 0, 0, 1) = 0.;
	convLayer.weights(0, 0, 1, 1) = 1.;
	convLayer.weights(0, 0, 2, 1) = 2.;
	convLayer.weights(0, 0, 0, 2) = 0.;
	convLayer.weights(0, 0, 1, 2) = 1.;
	convLayer.weights(0, 0, 2, 2) = 2.;
	convLayer.weights.dump4D(stdout, "Filter");
	convLayer.weights.moveToDevice();

	convLayer.bias.moveToHost();
	convLayer.bias(0) = 0.;
	convLayer.bias.dump(stdout, "Bias");
	convLayer.bias.moveToDevice();

	Tensor<float, 4> imageIn(Tensor<float, 4>::ON_CPU, { batchSize, inputChannels, imageWidth, imageHeight });
	imageIn(0, 0, 0, 0) = 0.;
	imageIn(0, 0, 1, 0) = 1.;
	imageIn(0, 0, 2, 0) = 2.;
	imageIn(0, 0, 3, 0) = 3.;
	imageIn(0, 0, 4, 0) = 4.;
	imageIn(0, 0, 0, 1) = 5.;
	imageIn(0, 0, 1, 1) = 6.;
	imageIn(0, 0, 2, 1) = 7.;
	imageIn(0, 0, 3, 1) = 8.;
	imageIn(0, 0, 4, 1) = 9.;
	imageIn(0, 0, 0, 2) = 10.;
	imageIn(0, 0, 1, 2) = 11.;
	imageIn(0, 0, 2, 2) = 12.;
	imageIn(0, 0, 3, 2) = 13.;
	imageIn(0, 0, 4, 2) = 14.;
	imageIn(0, 0, 0, 3) = 15.;
	imageIn(0, 0, 1, 3) = 16.;
	imageIn(0, 0, 2, 3) = 17.;
	imageIn(0, 0, 3, 3) = 18.;
	imageIn(0, 0, 4, 3) = 19.;
	imageIn(0, 0, 0, 4) = 20.;
	imageIn(0, 0, 1, 4) = 21.;
	imageIn(0, 0, 2, 4) = 22.;
	imageIn(0, 0, 3, 4) = 23.;
	imageIn(0, 0, 4, 4) = 24.;
	imageIn.dump4D(stdout, "Input Image");
	imageIn.moveToDevice();

	Tensor<float, 4> imageOut = convLayer.forward(imageIn);
	imageOut.moveToHost();
	imageOut.dump4D(stdout, "Output Image");

	Tensor<float, 4> error_tensor(Tensor<float, 4>::ON_CPU, { batchSize, outputChannels, imageWidth / strideX, imageHeight / strideY });
	error_tensor(0, 0, 0, 0) = 0.;
	error_tensor(0, 0, 1, 0) = 1.;
	error_tensor(0, 0, 0, 1) = 2.;
	error_tensor(0, 0, 1, 1) = 3.;
	error_tensor.dump4D(stdout, "Error Tensor");
	error_tensor.moveToDevice();

	convLayer.optimizer = new GPUSgd(0.001);

	Tensor<float, 4> next_error_tensor = convLayer.backward(error_tensor);
	next_error_tensor.moveToHost();
	next_error_tensor.dump4D(stdout, "Next Error Tensor");
#endif

	return EXIT_SUCCESS;
}

