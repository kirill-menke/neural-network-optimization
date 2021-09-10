#pragma once

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>

#include "./utils.h"

class MembraneLoader {
public:

#if __linux__
	static constexpr int NUM_IMAGES = 30;
	static constexpr int IMAGE_WIDTH = 512;
	static constexpr int IMAGE_HEIGHT = 512;
#else
	static constexpr int NUM_IMAGES = 60;
	static constexpr int IMAGE_WIDTH = 256;
	static constexpr int IMAGE_HEIGHT = 256;
#endif

	static constexpr float MEAN = 0.78030;
	static constexpr float STD = 0.41403;

	MembraneLoader(const char *dir, int batch_size):
		dir(dir),
		batch_size(batch_size) {}

	std::pair<Tensor<float, 4>, Tensor<float, 4>> loadBatch();
	float checkAccuracy(Tensor<float, 4> &pred, Tensor<float, 4> &truth);

private:
	const char *dir;
	int batch_size;
	int batch = 0;
	std::vector<std::pair<Tensor<float, 4>, Tensor<float, 4>>> batches;
};

class MNISTLoader {
public:
	static constexpr int NUM_IMAGES = 42000;
	static constexpr int IMAGE_WIDTH = 32; // 28;
	static constexpr int IMAGE_HEIGHT = 32; // 28;

	MNISTLoader(const char *filepath, int batch_size):
		file(filepath),
		batch_size(batch_size) {}

	std::pair<Tensor<float, 4>, Tensor<float, 4>> loadBatch();
	float checkAccuracy(Tensor<float, 4> &pred, Tensor<float, 4> &truth);

private:
	std::ifstream file;
	int batch_size;
	int line = 0;
};

