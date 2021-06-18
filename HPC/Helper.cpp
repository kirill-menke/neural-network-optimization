#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <algorithm>

#include "Helper.h"
#include <numeric>

void printTensor(Eigen::Tensor<float, 4> const &tensor) {
	auto dim = tensor.dimensions();
	for (int i = 0; i < dim[0]; i++) {
		std::cout << "[";
		for (int j = 0; j < dim[1]; j++) {
			std::cout << " [";
			for (int k = 0; k < dim[2]; k++) {
				for (int l = 0; l < dim[3]; l++) {
					std::cout << tensor(i, j, k, l);
					if (l != dim[3] - 1)
						std::cout << "\t";
				}
				if (k == dim[2] - 1)
					std::cout << "]";
				else
					std::cout << std::endl << "   ";
			}
			if (j != dim[1] - 1)
				std::cout << std::endl;
			std::cout << " ";
		}
		std::cout << "]" << std::endl;
	}
	std::cout << std::endl << std::endl;
}

void printTensor(Eigen::Tensor<float, 2> const &tensor) {
	auto dim = tensor.dimensions();
	std::cout << std::setprecision(5) << "  [";
	for (int k = 0; k < dim[0]; k++) {
		for (int l = 0; l < dim[1]; l++) {
			std::cout << tensor(k, l);
			if (l != dim[1] - 1)
				std::cout << "\t";
		}
		if (k == dim[0] - 1)
			std::cout << "]";
		else
			std::cout << std::endl << "   ";
	}
	std::cout << std::endl << std::endl;
}

std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>>
MNISTLoader::loadBatch(int batchSize)
{
	auto images = std::make_shared<Eigen::Tensor<float, 4>>(batchSize, 1, IMAGE_WIDTH, IMAGE_HEIGHT);
	auto numbers = std::make_shared<Eigen::Tensor<float, 2>>(batchSize, 10);
	numbers->setZero();

	for (int b = 0; b < batchSize; b++) {
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

		if (!file.good()) {
			fprintf(stderr, "MNISTLoader: fatal error!\n");
			exit(EXIT_FAILURE);
		}

		(*numbers)(b, digit) = 1.0f;

		for (int y = 0; y < IMAGE_HEIGHT; y++) {
			for (int x = 0; x < IMAGE_WIDTH; x++) {
				int pixel;
				file >> pixel;

				(*images)(b, 0, x, y) = normalize(pixel);
			}
		}
	}

	return { numbers, images };
}


void MNISTLoader::loadFullDataset() {
	images = Eigen::Tensor<float, 4>(IMAGE_NUM, 1, IMAGE_HEIGHT, IMAGE_WIDTH);
	numbers = Eigen::Tensor<float, 2>(IMAGE_NUM, 10);
	image_order = std::vector<int>(IMAGE_NUM);
	std::iota(image_order.begin(), image_order.end(), 0);

	std::string line;
	int cnt = 0;
	while (cnt < IMAGE_NUM)
	{
		std::getline(file, line);
		std::istringstream iss(line);
		int digit;
		iss >> digit;
		numbers(cnt, digit) = 1.0f;

		for (int y = 0; y < IMAGE_HEIGHT; y++) {
			for (int x = 0; x < IMAGE_WIDTH; x++) {
				int pixel;
				iss >> pixel;
				images(cnt, 0, y, x) = normalize(pixel);
			}
		}

		cnt++;
	}
}


std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>>
MNISTLoader::getBatch(int batchSize) {
	static int batch_count = 0;
	static int epoch = 1;
	
	if (batch_count + batchSize > IMAGE_NUM) {
		std::cout << "\n<<< Epoch " << ++epoch << " >>>" << std::endl;
		std::shuffle(image_order.begin(), image_order.end(), rand_gen);
		batch_count = 0;
	}

	// Create new tensor for data augmentation
	auto image_batch = std::make_shared<Eigen::Tensor<float, 4>>(batchSize, 1, IMAGE_HEIGHT, IMAGE_WIDTH);
	auto label_batch = std::make_shared<Eigen::Tensor<float, 2>>(batchSize, 10);

	for (int i = 0; i < batchSize; i++) {
		int img_idx = image_order[batch_count + i];
		
		for (int x = 0; x < IMAGE_HEIGHT; x++) {
			for (int y = 0; y < IMAGE_WIDTH; y++) {
				(*image_batch)(i, 0, x, y) = images(img_idx, 0, x, y);
			}
		}

		for (int j = 0; j < 10; j++) {
			(*label_batch)(i, j) = numbers(img_idx, j);
		}

	}

	augment(image_batch);

	batch_count += batchSize;

	return { label_batch, image_batch };
}


float MNISTLoader::normalize(float pixel) {
	return (pixel / 255 - mean / 255) / (std / 255);
}


void MNISTLoader::augment(std::shared_ptr<Eigen::Tensor<float, 4>> batch) {

}

