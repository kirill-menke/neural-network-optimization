#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>

void printTensor(Eigen::Tensor<float, 4> &tensor);
void printTensor(Eigen::Tensor<float, 2> &tensor);

class MNISTLoader {
public:
	static constexpr int IMAGE_WIDTH = 28;
	static constexpr int IMAGE_HEIGHT = 28;

	MNISTLoader(const char *filepath):
		file(filepath) {}

	/*
	 * Return a pair of the actual numbers in the images
	 * and a batch of images. Images have a channel-Dimension of
	 * 1 and a width/height of 28x28 pixels.
	 *
	 * TODO: Normalize!
	 */
	std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>> loadBatch(int batchSize);

private:
	std::ifstream file;
	int line = 0;
};

