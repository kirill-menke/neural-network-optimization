#include <stdio.h>
#include <stdlib.h>
#include <iomanip>

#include "Helper.h"

void printTensor(Eigen::Tensor<float, 4> &tensor) {
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

void printTensor(Eigen::Tensor<float, 2> &tensor) {
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

				(*images)(b, 0, x, y) = pixel;
			}
		}
	}

	return std::make_pair(numbers, images);
}

