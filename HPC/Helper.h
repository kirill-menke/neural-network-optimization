#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>

void printTensor(Eigen::Tensor<float, 4> const &tensor);
void printTensor(Eigen::Tensor<float, 2> const &tensor);

class MNISTLoader {
public:
	static constexpr int IMAGE_NUM = 42000;
	static constexpr int IMAGE_WIDTH = 28;
	static constexpr int IMAGE_HEIGHT = 28;

	static constexpr float mean = 33.40891f;
	static constexpr float std = 78.67774f;

	MNISTLoader(const char *filepath):
		file(filepath), rand_gen(0) {}

	/*
	 * Return a pair of the actual numbers in the images
	 * and a batch of images. Images have a channel-Dimension of
	 * 1 and a width/height of 28x28 pixels.
	 */
	std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>> loadBatch(int batchSize);

	/* Loads all images into RAM */
	void loadFullDataset(float testSize, bool shuffle);

	/* Returns batch and shuffles data after one epoch */
	std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>> getBatch(int batchSize);

	/* Returns test dataset */
	std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>> getTestSet();
	

private:
	std::ifstream file;
	int line = 0;

	std::mt19937_64 rand_gen;
	std::vector<int> train_order;

	Eigen::Tensor<float, 4> train_images;
	Eigen::Tensor<float, 2> train_labels;
	std::shared_ptr<Eigen::Tensor<float, 4>> test_images;
	std::shared_ptr<Eigen::Tensor<float, 2>> test_labels;


	/*
	 * Scales the data range down from [0, 255] to [0, 1].
	 * Then normalizes it such that mean = 0 and std = 1.
	 */
	float normalize(float pixel);


	/* Perform random data augmentation on batch */
	void augment(std::shared_ptr<Eigen::Tensor<float, 4>> batch);
};

