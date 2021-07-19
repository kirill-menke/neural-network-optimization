#include <cstdio>
#include <iomanip>
#include <string>
#include <random>

#include "utils.h"
#include "cuda-utils.h"

constexpr int MembraneLoader::NUM_IMAGES;
constexpr int MembraneLoader::IMAGE_WIDTH;
constexpr int MembraneLoader::IMAGE_HEIGHT;


std::shared_ptr<Eigen::Tensor<float, 4>> toEigenTensor(Tensor<float, 4> &tensor) {
	tensor.moveToHost();

	auto eigenTensor = std::make_shared<Eigen::Tensor<float, 4>>(
			tensor.dim(0), tensor.dim(1), tensor.dim(2), tensor.dim(3));
	for (int b = 0; b < tensor.dim(0); b++) {
		for (int c = 0; c < tensor.dim(1); c++) {
			for (int x = 0; x < tensor.dim(2); x++) {
				for (int y = 0; y < tensor.dim(3); y++) {
					(*eigenTensor)(b, c, x, y) = tensor(b, c, x, y);
				}
			}
		}
	}

	return eigenTensor;
}

Tensor<float, 4> fromEigenTensor(Eigen::Tensor<float, 4> &eigenTensor) {
	Tensor<float, 4> tensor({
		int(eigenTensor.dimension(0)),
		int(eigenTensor.dimension(1)),
		int(eigenTensor.dimension(2)),
		int(eigenTensor.dimension(3)) });

	for (int b = 0; b < tensor.dim(0); b++) {
		for (int c = 0; c < tensor.dim(1); c++) {
			for (int x = 0; x < tensor.dim(2); x++) {
				for (int y = 0; y < tensor.dim(3); y++) {
					tensor(b, c, x, y) = eigenTensor(b, c, x, y);
				}
			}
		}
	}

	tensor.moveToDevice();
	return tensor;
}

std::pair<Tensor<float, 4>, Tensor<float, 3>>
MembraneLoader::loadBatch(int batchSize) {
	Tensor<float, 4> images({ batchSize, 1, IMAGE_WIDTH, IMAGE_HEIGHT });
	Tensor<float, 3> labels({ batchSize, IMAGE_WIDTH, IMAGE_HEIGHT });

	for (int b = 0; b < batchSize; b++) {
		this->image = (this->image + 1) % NUM_IMAGES;

		std::ifstream image_file(std::string(this->dirPath)
			+ "/train-" + std::to_string(this->image) + ".png.txt");
	
		std::ifstream label_file(std::string(this->dirPath)
			+ "/label-" + std::to_string(this->image) + ".png.txt");

		for (int y = 0; y < IMAGE_HEIGHT; y++) {
			for (int x = 0; x < IMAGE_WIDTH; x++) {
				float pixel;
				image_file >> pixel;
				images(b, 0, x, y) = pixel;

				label_file >> pixel;
				labels(b, x, y) = pixel;
			}
		}
	}

	images.moveToDevice();
	labels.moveToDevice();
	return { images, labels };
}

static std::mt19937_64 rng(42);

void uniformRandomInit(float min, float max,
		Tensor<float, 4> &weights, Tensor<float, 1> &bias) {

	std::uniform_real_distribution<float> unif(min, max);

	for (int f = 0; f < weights.dim(0); f++)
		for (int c = 0; c < weights.dim(1); c++)
			for (int x = 0; x < weights.dim(2); x++)
				for (int y = 0; y < weights.dim(3); y++)
					weights(f, c, x, y) = unif(rng);

	for (int f = 0; f < bias.dim(0); f++)
		bias(f) = unif(rng);

	weights.moveToDevice();
	bias.moveToDevice();
}
