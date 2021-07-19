#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>

#include "./Tensor.h"

std::shared_ptr<Eigen::Tensor<float, 4>> toEigenTensor(Tensor<float, 4> &tensor);
Tensor<float, 4> fromEigenTensor(Eigen::Tensor<float, 4> &eigenTensor);

void uniformRandomInit(float min, float max, Tensor<float, 4> &weights, Tensor<float, 1> &bias);

class MembraneLoader {
	// TODO: Do some data augmentation, shuffle, etc.

public:
	static constexpr int NUM_IMAGES = 30;
	static constexpr int IMAGE_WIDTH = 512;
	static constexpr int IMAGE_HEIGHT = 512;

	MembraneLoader(const char *dirPath):
		dirPath(dirPath) {}

	/*
	 * The first return value is a image with a channel dimension of 1.
	 * The second return value is the labels for each pixel
	 * with hot-1-encoding. There are two classes (Membrane or not Membrane).
	 */
	std::pair<Tensor<float, 4>, Tensor<float, 3>> loadBatch(int batchSize);

private:
	const char *dirPath;
	int image = 0;
};

