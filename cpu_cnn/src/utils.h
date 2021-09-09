#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include <fstream>


class MNISTLoader {
public:

	MNISTLoader(const char* filepath);

	/* Loads only one batch into RAM: Returns a pair of labels (actual numbers represented in the images)
	 * and the images itself. Images have a channel-Dimension of 1 and a width/height of 28x28 pixels. */
	std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>> loadBatch(int batchSize);

	/* Loads all images and labels into RAM */
	void loadFullDataset(float testSize, bool shuffle);

	/* Returns a batch and shuffles data after one epoch */
	std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>> getBatch(int batchSize);

	/* Returns test dataset */
	std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>> getTestSet();
	

private:

	int image_num = 40000;
	int image_width = 28;
	int image_height = 28;

	float mean = 33.40891f;
	float std = 78.67774f;

	std::ifstream file;
	int line = 0;

	std::mt19937_64 rand_gen;
	std::vector<int> train_order;

	/* Train dataset */
	Eigen::Tensor<float, 4> train_images;
	Eigen::Tensor<float, 2> train_labels;

	/* Test dataset */
	std::shared_ptr<Eigen::Tensor<float, 4>> test_images;
	std::shared_ptr<Eigen::Tensor<float, 2>> test_labels;


	/*
	 * Scales the data range down from [0, 255] to [0, 1].
	 * Then normalizes it such that mean = 0 and std = 1.
	 */
	float normalize(float pixel);


	/* Performs random data augmentation on batch */
	void augment(std::shared_ptr<Eigen::Tensor<float, 4>> batch);
};


float calculateAccuracy(std::shared_ptr<Eigen::Tensor<float, 2>> truth, std::shared_ptr<Eigen::Tensor<float, 2>> prediction);

void printTensor(Eigen::Tensor<float, 4> const& tensor);
void printTensor(Eigen::Tensor<float, 2> const& tensor);

void assertSize(std::shared_ptr<Eigen::Tensor<float, 4>> tensor, int d1, int d2, int d3, int d4);
void assertSize(std::shared_ptr<Eigen::Tensor<float, 2>> tensor, int d1, int d2);

