#include "./data.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdio>

constexpr int MembraneLoader::NUM_IMAGES;
constexpr int MembraneLoader::IMAGE_WIDTH;
constexpr int MembraneLoader::IMAGE_HEIGHT;

std::pair<Tensor<float, 4>, Tensor<float, 4>> MembraneLoader::loadBatch() {
	if (batch < int(batches.size())) {
		auto data = batches[batch];
		batch = (batch + 1) % (NUM_IMAGES / batch_size);
		return data;
	}
	
	Tensor<float, 4> images(batch_size, 1, IMAGE_WIDTH, IMAGE_HEIGHT);
	Tensor<float, 4> labels(batch_size, 2, IMAGE_WIDTH, IMAGE_HEIGHT);

	for (int b = 0; b < batch_size; b++) {
		int image = (batch * batch_size + b) % NUM_IMAGES;
		// printf("Loading %d\n", image);

		std::ifstream image_file(std::string(dir)
			+ "/train-" + std::to_string(image) + ".png.txt");
	
		std::ifstream label_file(std::string(dir)
			+ "/label-" + std::to_string(image) + ".png.txt");

		if (image_file.fail() || label_file.fail()) {
			fprintf(stderr, "MembraneLoader: Failed to open image/label textfiles (idx=%d)!\n", image);
			exit(EXIT_FAILURE);
		}

		float pixel;
		for (int y = 0; y < IMAGE_HEIGHT; y++) {
			for (int x = 0; x < IMAGE_WIDTH; x++) {
				image_file >> pixel;
				assert(0. <= pixel && pixel <= 1.);
				images(b, 0, x, y) = (pixel / 255 - MEAN / 255) / (STD/ 255);;

				label_file >> pixel;
				assert(pixel == 1. || pixel == 0.);
				labels(b, 0, x, y) = pixel;
				labels(b, 1, x, y) = pixel == 0. ? 1. : 0.;
			}
		}
	}

	batches.push_back({ images, labels });
	images.moveToDevice();
	labels.moveToDevice();
	batch = (batch + 1) % (NUM_IMAGES / batch_size);
	return { images, labels };
}

float MembraneLoader::checkAccuracy(Tensor<float, 4> &pred, Tensor<float, 4> &truth) {
	pred.moveToHost();
	assert(pred.dim(0) == batch_size
		&& pred.dim(1) == 2
		&& pred.dim(2) == IMAGE_WIDTH
		&& pred.dim(3) == IMAGE_HEIGHT);

	double total_pixels = batch_size * IMAGE_WIDTH * IMAGE_HEIGHT;
	unsigned long long correct_pixels = 0.;

	for (int b = 0; b < batch_size; b++) {
		for (int x = 0; x < IMAGE_WIDTH; x++) {
			for (int y = 0; y < IMAGE_HEIGHT; y++) {
				bool predicted_class = pred(b, 0, x, y) > pred(b, 1, x, y);
				bool true_class = truth(b, 0, x, y) == 1.;
				if (predicted_class == true_class)
					correct_pixels += 1;
			}
		}
	}

	return (correct_pixels / total_pixels) * 100.;
}

constexpr int MNISTLoader::NUM_IMAGES;
constexpr int MNISTLoader::IMAGE_WIDTH;
constexpr int MNISTLoader::IMAGE_HEIGHT;

std::pair<Tensor<float, 4>, Tensor<float, 4>> MNISTLoader::loadBatch() {
	Tensor<float, 4> images(batch_size, 1, IMAGE_WIDTH, IMAGE_HEIGHT);
	Tensor<float, 4> labels(batch_size, 10, 1, 1);

	for (int b = 0; b < batch_size; b++) {
		int digit;
		file >> digit;
		if (file.eof()) {
			line = 0;
			file.clear();
			file.seekg(0);
			file >> digit;
		} else {
			line++;
		}

		assert(0 <= digit && digit < 10);
		if (!file.good()) {
			fprintf(stderr, "MNISTLoader: fatal error!\n");
			exit(EXIT_FAILURE);
		}

		for (int c = 0; c < 10; c++)
			labels(b, c, 0, 0) = digit == c ? 1. : 0.;

		for (int y = 0; y < IMAGE_HEIGHT; y++) {
			for (int x = 0; x < IMAGE_WIDTH; x++) {
				if (x >= 28 || y >= 28) {
					images(b, 0, x, y) = 0.;
					continue;
				}

				int pixel;
				file >> pixel;
				assert(0 <= pixel && pixel <= 255);
				images(b, 0, x, y) = float(pixel) / 255.0;
			}
		}
	}

	images.moveToDevice();
	labels.moveToDevice();
	return { images, labels };
}

float MNISTLoader::checkAccuracy(Tensor<float, 4> &pred, Tensor<float, 4> &truth) {	
	pred.moveToHost();
	assert(pred.dim(0) == batch_size
		&& pred.dim(1) == 10
		&& pred.dim(2) == 1
		&& pred.dim(3) == 1);

	unsigned long long int correct = 0;
	for (int b = 0; b < batch_size; b++) {
		int max_c = 0;
		float max = pred(b, 0, 0, 0);
		for (int c = 1; c < 10; c++) {
			float val = pred(b, c, 0, 0);
			if (val > max) {
				max = val;
				max_c = c;
			}
		}

		if (truth(b, max_c, 0, 0) == 1.)
			correct += 1;
	}

	return (float(correct) / batch_size) * 100.0;
}

