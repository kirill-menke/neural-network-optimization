#include <iomanip>
#include <numeric>
#include <iostream>

#include "utils.h"


MNISTLoader::MNISTLoader(const char* filepath) : file(filepath), rand_gen(0) {}


std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>>
MNISTLoader::loadBatch(int batchSize)
{
	auto images = std::make_shared<Eigen::Tensor<float, 4>>(batchSize, 1, image_width, image_height);
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

		for (int y = 0; y < image_height; y++) {
			for (int x = 0; x < image_width; x++) {
				int pixel;
				file >> pixel;

				(*images)(b, 0, x, y) = normalize(pixel);
			}
		}
	}

	return { numbers, images };
}


void MNISTLoader::loadFullDataset(float testSize, bool shuffle) {
	int test_num = static_cast<int>(testSize * image_num);
	int train_num = image_num - test_num;

	train_images = Eigen::Tensor<float, 4>(train_num, 1, image_height, image_width);
	train_labels = Eigen::Tensor<float, 2>(train_num, 10);
	test_images = std::make_shared<Eigen::Tensor<float, 4>>(test_num, 1, image_height, image_width);
	test_labels = std::make_shared<Eigen::Tensor<float, 2>>(test_num, 10);

	train_order = std::vector<int>(train_num);
	std::iota(train_order.begin(), train_order.end(), 0);
	
	// First load data, then shuffle, and then assign to train/test tensors
	Eigen::Tensor<float, 4> all_images(image_num, 1, image_height, image_width);
	Eigen::Tensor<float, 2> all_labels(image_num, 10);
	
	std::string line;
	for (int i = 0; i < image_num; i++) {
		std::getline(file, line);
		std::istringstream iss(line);
		int digit;
		iss >> digit;
		all_labels(i, digit) = 1.0f;

		for (int y = 0; y < image_height; y++) {
			for (int x = 0; x < image_width; x++) {
				int pixel;
				iss >> pixel;
				all_images(i, 0, y, x) = normalize(pixel);
			}
		}
	}

	std::vector<int> order(image_num);
	std::iota(order.begin(), order.end(), 0);
	
	if (shuffle) {
		std::shuffle(order.begin(), order.end(), rand_gen);
	}

	for (int i = 0; i < train_num; i++) {
		int idx = order[i];
		for (int y = 0; y < image_height; y++) {
			for (int x = 0; x < image_width; x++) {
				train_images(i, 0, y, x) = all_images(idx, 0, y, x);
			}
		}

		for (int j = 0; j < 10; j++) {
			train_labels(i, j) = all_labels(idx, j);
		}
	}

	for (int i = 0; i < test_num; i++) {
		int idx = order[train_num + i];

		for (int y = 0; y < image_height; y++) {
			for (int x = 0; x < image_width; x++) {
				(*test_images)(i, 0, y, x) = all_images(idx, 0, y, x);
			}
		}

		for (int j = 0; j < 10; j++) {
			(*test_labels)(i, j) = all_labels(idx, j);
		}
	}
}


std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>>
MNISTLoader::getBatch(int batchSize) {
	static int batch_count = 0;
	static int epoch = 1;
	
	if (batch_count + batchSize > train_images.dimension(0)) {
		std::cout << "\n<<< Epoch " << ++epoch << " >>>" << std::endl;
		std::shuffle(train_order.begin(), train_order.end(), rand_gen);
		batch_count = 0;
	}

	// Create new tensor for data augmentation
	auto image_batch = std::make_shared<Eigen::Tensor<float, 4>>(batchSize, 1, image_height, image_width);
	auto label_batch = std::make_shared<Eigen::Tensor<float, 2>>(batchSize, 10);

	for (int i = 0; i < batchSize; i++) {
		int img_idx = train_order[batch_count + i];
		
		for (int x = 0; x < image_height; x++) {
			for (int y = 0; y < image_width; y++) {
				(*image_batch)(i, 0, x, y) = train_images(img_idx, 0, x, y);
			}
		}

		for (int j = 0; j < 10; j++) {
			(*label_batch)(i, j) = train_labels(img_idx, j);
		}
	}

	augment(image_batch);

	batch_count += batchSize;

	return { label_batch, image_batch };
}


std::pair<std::shared_ptr<Eigen::Tensor<float, 2>>, std::shared_ptr<Eigen::Tensor<float, 4>>>
MNISTLoader::getTestSet() {
	return { test_labels, test_images };
}


float MNISTLoader::normalize(float pixel) {
	return (pixel / 255 - mean / 255) / (std / 255);
}


void MNISTLoader::augment(std::shared_ptr<Eigen::Tensor<float, 4>> batch) {

}


float calculateAccuracy(std::shared_ptr<Eigen::Tensor<float, 2>> truth, std::shared_ptr<Eigen::Tensor<float, 2>> prediction) {
	int batch_size = truth->dimension(0);
	int classes = truth->dimension(1);
	int correct = 0;
	for (int b = 0; b < batch_size; b++) {
		int maxidx = 0;
		for (int c = 0; c < classes; c++) {
			if ((*prediction)(b, c) > (*prediction)(b, maxidx)) {
				maxidx = c;
			}
		}

		if ((*truth)(b, maxidx) > 0.0) {
			correct += 1;
		}
	}

	return (float(correct) / float(batch_size)) * 100.f;
}


void assertSize(std::shared_ptr<Eigen::Tensor<float, 4>> tensor, int d1, int d2, int d3, int d4) {
	assert(tensor->dimension(0) == d1);
	assert(tensor->dimension(1) == d2);
	assert(tensor->dimension(2) == d3);
	assert(tensor->dimension(3) == d4);
}

void assertSize(std::shared_ptr<Eigen::Tensor<float, 2>> tensor, int d1, int d2) {
	assert(tensor->dimension(0) == d1);
	assert(tensor->dimension(1) == d2);
}

void printTensor(Eigen::Tensor<float, 4> const& tensor) {
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

void printTensor(Eigen::Tensor<float, 2> const& tensor) {
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

