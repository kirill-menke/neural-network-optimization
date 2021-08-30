#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <utility>
#include <string>
#include <cstring>
#include <cmath>
#include <cuda.h>

extern "C" {
#ifdef _LINUX
	#include <unistd.h>
#else
	#include <io.h>
	#define F_OK    0
#endif
}

#include "./tensor.h"
#include "./activations.h"
#include "./conv.h"
#include "./optimizer.h"
#include "./loss.h"
#include "./pooling.h"
#include "./utils.h"
#include "./data.h"

static float sum(const Tensor<float, 2> &t) {
	t.moveToHost();
	double sum = 0.;
	for (int x = 0; x < t.dim(0); x++)
		for (int y = 0; y < t.dim(1); y++)
			sum += t(x, y);

	return sum;
}

static std::pair<Tensor<float, 4>, Tensor<float, 4>> testInput(int batch_size, int image_size) {
	Tensor<float, 4> images(batch_size, 1, image_size, image_size);
	Tensor<float, 4> truths(batch_size, 2, image_size, image_size);

	srand(1234);

	for (int b = 0; b < batch_size; b++) {
		for (int x = 0; x < image_size; x++) {
			for (int y = 0; y < image_size; y++) {
#if 0
				bool t = x + y == image_size;
				images(b, 0, x, y) = t ? 0. : 1.;
				truths(b, 0, x, y) = t ? 1. : 0.;
				truths(b, 1, x, y) = t ? 0. : 1.;
#endif
#if 0
				bool t = rand() % 3 == 1;
				images(b, 0, x, y) = t ? 0. : 1.;
				truths(b, 0, x, y) = t ? 1. : 0.;
				truths(b, 1, x, y) = t ? 0. : 1.;
#endif
				bool t = (rand() & 0b011) == 0;
				images(b, 0, x, y) = t ? -0.5 : 0.5;
				truths(b, 0, x, y) = t ? 1. : 0.;
				truths(b, 1, x, y) = t ? 0. : 1.;
			}
		}
	}

	images.moveToDevice();
	truths.moveToDevice();
	return { images, truths };
}

static void bench_conv(int iterations) {
	float learing_rate = 0.000001;
	int batch_size = 5,
	    image_size = 256;

	Conv conv1(1, 32, 3);
	ReLU relu;

	Conv conv2(32, 2, 3);
	SoftMax softmax;

	PerPixelCELoss loss;
	conv1.optimizer = new Sgd(learing_rate);
	conv2.optimizer = new Sgd(learing_rate);
	uniformRandomInit(-0.001, 0.001, conv1.weights, conv1.bias);
	uniformRandomInit(-0.001, 0.001, conv2.weights, conv2.bias);

	float elapsed,
	      elapsed_time_forward = 0.,
	      elapsed_time_backward = 0.;
	cudaEvent_t start, stop;
	cudaErrchk(cudaEventCreate(&start));
	cudaErrchk(cudaEventCreate(&stop));

	auto data = testInput(batch_size, image_size);
	for (int i = 0; i < iterations; i++) {
		cudaErrchk(cudaEventRecord(start, 0));
		auto t1 = conv1.forward(data.first);
		cudaErrchk(cudaEventRecord(stop, 0));
		cudaErrchk(cudaEventSynchronize(stop));
		cudaErrchk(cudaEventElapsedTime(&elapsed, start, stop));
		elapsed_time_forward += elapsed;

		auto t2 = relu.forward(t1);

		cudaErrchk(cudaEventRecord(start, 0));
		auto t3 = conv2.forward(t2);
		cudaErrchk(cudaEventRecord(stop, 0));
		cudaErrchk(cudaEventSynchronize(stop));
		cudaErrchk(cudaEventElapsedTime(&elapsed, start, stop));
		elapsed_time_forward += elapsed;

		auto pred = softmax.forward(t3);
#if 0
		delete conv1.padded_input;
		delete relu.output_tensor;
		delete conv2.padded_input;
		delete softmax.output_tensor;
#else
		if (i % 10 == 0) {
			auto l = loss.forward(pred, data.second);
			printf("iter. %d:\tloss=%f\n", i, sum(l));
		}

		auto e1 = loss.backward(pred, data.second);
		auto e2 = softmax.backward(e1);

		cudaErrchk(cudaEventRecord(start, 0));
		auto e3 = conv2.backward(e2);
		cudaErrchk(cudaEventRecord(stop, 0));
		cudaErrchk(cudaEventSynchronize(stop));
		cudaErrchk(cudaEventElapsedTime(&elapsed, start, stop));
		elapsed_time_backward += elapsed;

		auto e4 = relu.backward(e3);

		cudaErrchk(cudaEventRecord(start, 0));
		auto e5 = conv1.backward(e4);
		cudaErrchk(cudaEventRecord(stop, 0));
		cudaErrchk(cudaEventSynchronize(stop));
		cudaErrchk(cudaEventElapsedTime(&elapsed, start, stop));
		elapsed_time_backward += elapsed;
#endif
	}

	cudaErrchk(cudaEventDestroy(start));
	cudaErrchk(cudaEventDestroy(stop));
	printf("conv. forward:  %.3fms\n", elapsed_time_forward);
	printf("conv. backward: %.3fms\n", elapsed_time_backward);
}

static void bench_mini_unet(int iterations) {
	int batch_size = 1;
	float learning_rate = 1e-6,
	      elapsed,
	      elapsed_conv_forward = 0.,
	      elapsed_conv_backward = 0.,
	      elapsed_activations_forward = 0.,
	      elapsed_activations_backward = 0.,
	      elapsed_maxpool_forward = 0.,
	      elapsed_maxpool_backward = 0.,
	      elapsed_upsample_forward = 0.,
	      elapsed_upsample_backward = 0.,
	      elapsed_split = 0.,
	      elapsed_concat = 0.;

	MembraneLoader dataloader("../data/cell-membranes/", batch_size);

	cudaEvent_t start, stop;
	cudaErrchk(cudaEventCreate(&start));
	cudaErrchk(cudaEventCreate(&stop));

	auto time_start = [&]() -> void {
		cudaErrchk(cudaPeekAtLastError());
		cudaErrchk(cudaEventRecord(start, 0));
	};

	auto time_end = [&](float &t) -> void {
		cudaErrchk(cudaEventRecord(stop, 0));
		cudaErrchk(cudaEventSynchronize(stop));
		cudaErrchk(cudaEventElapsedTime(&elapsed, start, stop));
		cudaErrchk(cudaDeviceSynchronize());
		t += elapsed;
	};

	Conv conv1(1, 20, 3);
	ReLU relu1;
	MaxPool maxpool1(2);

	Conv conv2(20, 40, 3);
	ReLU relu2;
	MaxPool maxpool2(2);

	Conv conv3(40, 80, 3);
	ReLU relu3;
	Upsample upsample3(2);

	Conv conv4(80 + 40, 80, 3);
	ReLU relu4;
	Upsample upsample4(2);

	Conv conv5(80 + 20, 2, 3);
	SoftMax softmax;

	PerPixelCELoss loss;
	float init = 0.01;
	conv1.optimizer = new Sgd(learning_rate);
	conv2.optimizer = new Sgd(learning_rate);
	conv3.optimizer = new Sgd(learning_rate);
	conv4.optimizer = new Sgd(learning_rate);
	conv5.optimizer = new Sgd(learning_rate);
	uniformRandomInit(-init, init, conv1.weights, conv1.bias);
	uniformRandomInit(-init, init, conv2.weights, conv2.bias);
	uniformRandomInit(-init, init, conv3.weights, conv3.bias);
	uniformRandomInit(-init, init, conv4.weights, conv4.bias);
	uniformRandomInit(-init, init, conv5.weights, conv5.bias);

	auto data = dataloader.loadBatch();
	// auto data = testInput(batch_size, image_size);
	for (int iter = 0; iter < iterations; iter++) {
		// auto data = dataloader.loadBatch();

		time_start();
		auto forward_conv1 = conv1.forward(data.first);
		time_end(elapsed_conv_forward);
		time_start();
		auto forward_relu1 = relu1.forward(forward_conv1);
		time_end(elapsed_activations_forward);
		time_start();
		auto forward_maxpool1 = maxpool1.forward(forward_relu1);
		time_end(elapsed_maxpool_forward);

		time_start();
		auto forward_conv2 = conv2.forward(forward_maxpool1);
		time_end(elapsed_conv_forward);
		time_start();
		auto forward_relu2 = relu2.forward(forward_conv2);
		time_end(elapsed_activations_forward);
		time_start();
		auto forward_maxpool2 = maxpool2.forward(forward_relu2);
		time_end(elapsed_maxpool_forward);

		time_start();
		auto forward_conv3 = conv3.forward(forward_maxpool2);
		time_end(elapsed_conv_forward);
		time_start();
		auto forward_relu3 = relu3.forward(forward_conv3);
		time_end(elapsed_activations_forward);
		time_start();
		auto forward_upsample3 = upsample3.forward(forward_relu3);
		time_end(elapsed_upsample_forward);

		time_start();
		auto forward_concat4 = concat(forward_upsample3, forward_relu2);
		time_end(elapsed_concat);
		time_start();
		auto forward_conv4 = conv4.forward(forward_concat4);
		time_end(elapsed_conv_forward);
		time_start();
		auto forward_relu4 = relu4.forward(forward_conv4);
		time_end(elapsed_activations_forward);
		time_start();
		auto forward_upsample4 = upsample4.forward(forward_relu4);
		time_end(elapsed_upsample_forward);

		time_start();
		auto forward_concat5 = concat(forward_upsample4, forward_relu1);
		time_end(elapsed_concat);
		time_start();
		auto forward_conv5 = conv5.forward(forward_concat5);
		time_end(elapsed_conv_forward);
		time_start();
		auto forward_softmax = softmax.forward(forward_conv5);
		time_end(elapsed_activations_forward);

		if (iter < 6 || iter % 10 == 0) {
			auto l = loss.forward(forward_softmax, data.second);
			if (iter < 6 || iter % 50 == 0)
				printf("iter. %d:\tloss=%f, correct_pixels=%f%%\n",
					iter, sum(l), dataloader.checkAccuracy(forward_softmax, data.second));
			else
				printf("iter. %d:\tloss=%f\n", iter, sum(l));
		}

		auto backward_loss = loss.backward(forward_softmax, data.second);
		time_start();
		auto backward_softmax = softmax.backward(backward_loss);
		time_end(elapsed_activations_backward);
		time_start();
		auto backward_conv5 = conv5.backward(backward_softmax);
		time_end(elapsed_conv_backward);
		time_start();
		auto backward_split5 = split(backward_conv5, forward_upsample4.dim(1));
		time_end(elapsed_split);

		time_start();
		auto backward_upsample4 = upsample4.backward(backward_split5.first);
		time_end(elapsed_upsample_backward);
		time_start();
		auto backward_relu4 = relu4.backward(backward_upsample4);
		time_end(elapsed_activations_backward);
		time_start();
		auto backward_conv4 = conv4.backward(backward_relu4);
		time_end(elapsed_conv_backward);
		time_start();
		auto backward_split4 = split(backward_conv4, forward_upsample3.dim(1));
		time_end(elapsed_split);

		time_start();
		auto backward_upsample3 = upsample3.backward(backward_split4.first);
		time_end(elapsed_upsample_backward);
		time_start();
		auto backward_relu3 = relu3.backward(backward_upsample3);
		time_end(elapsed_activations_backward);
		time_start();
		auto backward_conv3 = conv3.backward(backward_relu3);
		time_end(elapsed_conv_backward);

		time_start();
		auto backward_maxpool2 = maxpool2.backward(backward_conv3);
		time_end(elapsed_maxpool_backward);
		auto backward_add2 = backward_maxpool2 + backward_split4.second;
		time_start();
		auto backward_relu2 = relu2.backward(backward_add2);
		time_end(elapsed_activations_backward);
		time_start();
		auto backward_conv2 = conv2.backward(backward_relu2);
		time_end(elapsed_conv_backward);

		time_start();
		auto backward_maxpool1 = maxpool1.backward(backward_conv2);
		time_end(elapsed_maxpool_backward);
		auto backward_add1 = backward_maxpool1 + backward_split5.second;
		time_start();
		auto backward_relu1 = relu1.backward(backward_add1);
		time_end(elapsed_activations_backward);
		time_start();
		auto backward_conv1 = conv1.backward(backward_relu1);
		time_end(elapsed_conv_backward);

#if 1
		if (iter != 0 && iter % 100 == 0) {
			learning_rate *= 0.5;
			printf("iter. %d:\tlearning_rate: %e\n", iter, learning_rate);
			conv1.optimizer->learning_rate = learning_rate;
			conv2.optimizer->learning_rate = learning_rate;
			conv3.optimizer->learning_rate = learning_rate;
			conv4.optimizer->learning_rate = learning_rate;
			conv5.optimizer->learning_rate = learning_rate;
		}
#endif
	}

	cudaErrchk(cudaEventDestroy(start));
	cudaErrchk(cudaEventDestroy(stop));
	printf("convolution forward:   %.3fms\n", elapsed_conv_forward / 5.);
	printf("convolution backward:  %.3fms\n", elapsed_conv_backward / 5.);
	printf("relu/softmax forward:  %.3fms\n", elapsed_activations_forward / 5.);
	printf("relu/softmax backward: %.3fms\n", elapsed_activations_backward / 5.);
	printf("maxpool forward:       %.3fms\n", elapsed_maxpool_forward / 2.);
	printf("maxpool backward:      %.3fms\n", elapsed_maxpool_backward / 2.);
	printf("upsample forward:      %.3fms\n", elapsed_upsample_forward / 2.);
	printf("upsample backward:     %.3fms\n", elapsed_upsample_backward / 2.);
	printf("split:                 %.3fms\n", elapsed_split / 2.);
	printf("concat:                %.3fms\n", elapsed_concat / 2.);
}

static void bench_mnist(int iterations) {
	int batch_size = 15;
	float learning_rate = 0.01;

	/* 32x32 */
	Conv conv1(1, 16, 3);
	ReLU relu1;
	MaxPool pool1(2);

	/* 16x16 */
	Conv conv2(16, 32, 3);
	ReLU relu2;
	MaxPool pool2(2);

	/* 8x8 */
	Conv conv3(32, 64, 3);
	ReLU relu3;
	MaxPool pool3(8);

	/*1x1*/
	Conv conv4(64, 10, 1);
	SoftMax softmax;

	PerPixelCELoss loss;
	conv1.optimizer = new Sgd(learning_rate);
	conv2.optimizer = new Sgd(learning_rate);
	conv3.optimizer = new Sgd(learning_rate);
	conv4.optimizer = new Sgd(learning_rate);
	uniformRandomInit(-0.05, 0.05, conv1.weights, conv1.bias);
	uniformRandomInit(-0.05, 0.05, conv2.weights, conv2.bias);
	uniformRandomInit(-0.05, 0.05, conv3.weights, conv3.bias);
	uniformRandomInit(-0.05, 0.05, conv4.weights, conv4.bias);

	MNISTLoader mnist("../data/mnist-train.txt", batch_size);
	auto data = mnist.loadBatch();
	for (int iter = 0; iter < iterations; iter++) {
		// auto data = mnist.loadBatch();

		auto t1 = pool1.forward(relu1.forward(conv1.forward(data.first)));
		auto t2 = pool2.forward(relu2.forward(conv2.forward(t1)));
		auto t3 = pool3.forward(relu3.forward(conv3.forward(t2)));
		auto pred = softmax.forward(conv4.forward(t3));

		if (iter % 20 == 0) {
			float l = sum(loss.forward(pred, data.second));
			printf("iter. %d:\tloss=%f, correct_digits=%.1f%%\n", iter, l, mnist.checkAccuracy(pred, data.second));
			// pred.dump4D(stdout, "prediction");
		}

		auto e = loss.backward(pred, data.second);
		auto e4 = conv4.backward(softmax.backward(e));
		auto e3 = conv3.backward(relu3.backward(pool3.backward(e4)));
		auto e2 = conv2.backward(relu2.backward(pool2.backward(e3)));
		auto e1 = conv1.backward(relu1.backward(pool1.backward(e2)));
	}
}

static void test_unet(int iterations) {
	int batch_size = 2;
	float learning_rate = 1e-3;

	Conv conv_0_1(1, 2, 3);
	ReLU relu_0_1;
	Conv conv_0_2(2, 4, 3);
	ReLU relu_0_2;
	MaxPool pool_0(2);

	Conv conv_1_1(4, 8, 3);
	ReLU relu_1_1;
	MaxPool pool_1(2);

	Conv conv_2_1(8, 16, 3);
	ReLU relu_2_1;
	MaxPool pool_2(2);

	Conv conv_3_1(16, 16, 3);
	ReLU relu_3_1;
	Conv conv_3_2(16, 8, 3);
	ReLU relu_3_2;
	Upsample upsample_3(2);

	// concat(upsample_3, relu_2_1)
	Conv conv_4_1(8 + 16, 8, 3);
	ReLU relu_4_1;
	Upsample upsample_4(2);

	// concat(upsample_4, relu_1_1)
	Conv conv_5_1(8 + 8, 8, 3);
	ReLU relu_5_1;
	Upsample upsample_5(2);

	// concat(upsample_5, relu_0_1)
	Conv conv_6_1(8 + 4, 4, 3);
	ReLU relu_6_1;
	Conv conv_6_2(4, 2, 1);
	SoftMax softmax;

	PerPixelCELoss loss;

	conv_0_1.optimizer = new Adam(learning_rate, 0.9, 0.999, std::tuple<int, int, int, int>{2, 1, 256, 256});
	conv_0_2.optimizer = new Adam(learning_rate, 0.9, 0.999, std::tuple<int, int, int, int>{4, 2, 256, 256});
	conv_1_1.optimizer = new Adam(learning_rate, 0.9, 0.999, std::tuple<int, int, int, int>{8, 4, 256, 256});
	conv_2_1.optimizer = new Adam(learning_rate, 0.9, 0.999, std::tuple<int, int, int, int>{16, 8, 256, 256});
	conv_3_1.optimizer = new Adam(learning_rate, 0.9, 0.999, std::tuple<int, int, int, int>{16, 16, 256, 256});
	conv_3_2.optimizer = new Adam(learning_rate, 0.9, 0.999, std::tuple<int, int, int, int>{8, 16, 256, 256});
	conv_4_1.optimizer = new Adam(learning_rate, 0.9, 0.999, std::tuple<int, int, int, int>{8, 8 + 16, 256, 256});
	conv_5_1.optimizer = new Adam(learning_rate, 0.9, 0.999, std::tuple<int, int, int, int>{8, 8 + 8, 256, 256});
	conv_6_1.optimizer = new Adam(learning_rate, 0.9, 0.999, std::tuple<int, int, int, int>{4, 8 + 4, 256, 256});
	conv_6_2.optimizer = new Adam(learning_rate, 0.9, 0.999, std::tuple<int, int, int, int>{2, 4, 256, 256});


	if (_access("./weights/unet/conv-0-1.weights", F_OK) != 0) {
		float init = 0.5;
		printf("Initializing weights from %f to %f...\n", -init, init);
		uniformRandomInit(-init, init, conv_0_1.weights, conv_0_1.bias);
		uniformRandomInit(-init, init, conv_0_2.weights, conv_0_2.bias);
		uniformRandomInit(-init, init, conv_1_1.weights, conv_1_1.bias);
		uniformRandomInit(-init, init, conv_2_1.weights, conv_2_1.bias);
		uniformRandomInit(-init, init, conv_3_1.weights, conv_3_1.bias);
		uniformRandomInit(-init, init, conv_3_2.weights, conv_3_2.bias);
		uniformRandomInit(-init, init, conv_4_1.weights, conv_4_1.bias);
		uniformRandomInit(-init, init, conv_5_1.weights, conv_5_1.bias);
		uniformRandomInit(-init, init, conv_6_1.weights, conv_6_1.bias);
		uniformRandomInit(-init, init, conv_6_2.weights, conv_6_2.bias);
	} else {
		printf("Reading weights...\n");
		readFromFile("./weights/unet/conv-0-1.weights", conv_0_1.weights, conv_0_1.bias);
		readFromFile("./weights/unet/conv-0-2.weights", conv_0_2.weights, conv_0_2.bias);
		readFromFile("./weights/unet/conv-1-1.weights", conv_1_1.weights, conv_1_1.bias);
		readFromFile("./weights/unet/conv-2-1.weights", conv_2_1.weights, conv_2_1.bias);
		readFromFile("./weights/unet/conv-3-1.weights", conv_3_1.weights, conv_3_1.bias);
		readFromFile("./weights/unet/conv-3-2.weights", conv_3_2.weights, conv_3_2.bias);
		readFromFile("./weights/unet/conv-4-1.weights", conv_4_1.weights, conv_4_1.bias);
		readFromFile("./weights/unet/conv-5-1.weights", conv_5_1.weights, conv_5_1.bias);
		readFromFile("./weights/unet/conv-6-1.weights", conv_6_1.weights, conv_6_1.bias);
		readFromFile("./weights/unet/conv-6-2.weights", conv_6_2.weights, conv_6_2.bias);
	}

	MembraneLoader dataloader("../data/cell-membranes/", batch_size);

	printf("Start iterations...\n");
	for (int iter = 0; iter < iterations; iter++) {
		auto data = dataloader.loadBatch();

		auto t0 = relu_0_2.forward(
			conv_0_2.forward(
			relu_0_1.forward(
			conv_0_1.forward(data.first))));
		auto t1 = relu_1_1.forward(conv_1_1.forward(pool_0.forward(t0)));
		auto t2 = relu_2_1.forward(conv_2_1.forward(pool_1.forward(t1)));
		auto t3 = relu_3_2.forward(
			conv_3_2.forward(
			relu_3_1.forward(
			conv_3_1.forward(pool_2.forward(t2)))));
		auto t4in = concat(upsample_3.forward(t3), t2);
		auto t4 = relu_4_1.forward(conv_4_1.forward(t4in));
		auto t5in = concat(upsample_4.forward(t4), t1);
		auto t5 = relu_5_1.forward(conv_5_1.forward(t5in));
		auto t6in = concat(upsample_5.forward(t5), t0);
		auto pred = softmax.forward(
			conv_6_2.forward(
			relu_6_1.forward(
			conv_6_1.forward(t6in))));

		auto l = sum(loss.forward(pred, data.second));
		printf("iter. %d:\tloss=%f\n", iter, l);
		if (iter % 10 == 0)
			printf("iter. %d:\tcorrect_pixels=%.2f\n", iter, dataloader.checkAccuracy(pred, data.second));

		auto e6 = conv_6_1.backward(
			relu_6_1.backward(
			conv_6_2.backward(
			softmax.backward(
			loss.backward(pred, data.second)))));
		auto e6split = split(e6, t5.dim(1));
		auto e5 = conv_5_1.backward(
			relu_5_1.backward(
			upsample_5.backward(e6split.first)));
		auto e5split = split(e5, t4.dim(1));
		auto e4 = conv_4_1.backward(
			relu_4_1.backward(
			upsample_4.backward(e5split.first)));
		auto e4split = split(e4, t3.dim(1));
		auto e3 = conv_3_1.backward(
			relu_3_1.backward(
			conv_3_2.backward(
			relu_3_2.backward(
			upsample_3.backward(e4split.first)))));
		auto e2 = conv_2_1.backward(
			relu_2_1.backward(
			pool_2.backward(e3) + e4split.second));
		auto e1 = conv_1_1.backward(
			relu_1_1.backward(
			pool_1.backward(e2) + e5split.second));
		auto e0 = conv_0_1.backward(
			relu_0_1.backward(
			conv_0_2.backward(
			relu_0_2.backward(
			pool_0.backward(e1) + e6split.second))));
	}

	printf("Done! (Saving weights...)\n");
	writeToFile("./weights/unet/conv-0-1.weights", conv_0_1.weights, conv_0_1.bias);
	writeToFile("./weights/unet/conv-0-2.weights", conv_0_2.weights, conv_0_2.bias);
	writeToFile("./weights/unet/conv-1-1.weights", conv_1_1.weights, conv_1_1.bias);
	writeToFile("./weights/unet/conv-2-1.weights", conv_2_1.weights, conv_2_1.bias);
	writeToFile("./weights/unet/conv-3-1.weights", conv_3_1.weights, conv_3_1.bias);
	writeToFile("./weights/unet/conv-3-2.weights", conv_3_2.weights, conv_3_2.bias);
	writeToFile("./weights/unet/conv-4-1.weights", conv_4_1.weights, conv_4_1.bias);
	writeToFile("./weights/unet/conv-5-1.weights", conv_5_1.weights, conv_5_1.bias);
	writeToFile("./weights/unet/conv-6-1.weights", conv_6_1.weights, conv_6_1.bias);
	writeToFile("./weights/unet/conv-6-2.weights", conv_6_2.weights, conv_6_2.bias);
}

int main(int argc, const char *argv[]) {
	cudaErrchk(cudaDeviceSynchronize());
	printf("Device synchronized!\n");
	if (argc == 1) {
		bench_mini_unet(1);
		return EXIT_SUCCESS;
	}

	if (argc != 3) {
		fprintf(stderr, "usage: %s <bench_unet|test_unet|mnist|bench_conv> <iterations>\n", argv[0]);
		return EXIT_FAILURE;
	}

	int iterations = atoi(argv[2]);
	if (strcmp(argv[1], "bench_unet") == 0)
		bench_mini_unet(iterations);
	else if (strcmp(argv[1], "test_unet") == 0)
		test_unet(iterations);
	else if (strcmp(argv[1], "mnist") == 0)
		bench_mnist(iterations);
	else if (strcmp(argv[1], "bench_conv") == 0)
		bench_conv(iterations);
	else
		fprintf(stderr, "usage: %s <bench_unet|test_unet|mnist|bench_conv> <iterations>\n", argv[0]);

	return EXIT_SUCCESS;
}

