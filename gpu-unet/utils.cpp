#include <random>
#include <cstdio>
#include <cstdlib>

#include "utils.h"

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
		bias(f) = 0.; // unif(rng);

	weights.moveToDevice();
	bias.moveToDevice();
}

static void die(const char msg[]) {
	perror(msg);
	exit(EXIT_FAILURE);
}

void writeToFile(const std::string &filename, const Tensor<float, 4> &weights, const Tensor<float, 1> &bias) {
	int output_channels = weights.dim(0),
	    input_channels = weights.dim(1),
	    filter_width = weights.dim(2),
	    filter_height = weights.dim(3);
	assert(bias.dim(0) == output_channels);
	weights.moveToHost();
	bias.moveToHost();

	FILE *f = fopen(filename.c_str(), "w");
	if (f == NULL)
		die(filename.c_str());

	size_t n;
	fprintf(f, "%d %d %d %d\n", output_channels, input_channels, filter_width, filter_height);
	n = fwrite((void *)weights.get_data(), sizeof(float), weights.num_elements(), f);
	if (n != weights.num_elements())
		die(("fwrite weights: " + filename + " n=" + std::to_string(n)).c_str());

	n = fwrite((void *)bias.get_data(), sizeof(float), bias.num_elements(), f);
	if (n != bias.num_elements())
		die(("fwrite bias: " + filename + " n=" + std::to_string(n)).c_str());

	if (ferror(f))
		die(filename.c_str());
	if (fclose(f) != 0)
		die(filename.c_str());
}

void readFromFile(const std::string &filename, Tensor<float, 4> &weights, Tensor<float, 1> &bias) {
	assert(bias.dim(0) == weights.dim(0));
	int output_channels, input_channels, filter_width, filter_height;

	FILE *f = fopen(filename.c_str(), "r");
	if (f == NULL)
		die(filename.c_str());

	if (fscanf(f, "%d %d %d %d\n", &output_channels, &input_channels, &filter_width, &filter_height) != 4)
		die(filename.c_str());

	assert(output_channels == weights.dim(0)
		&& input_channels == weights.dim(1)
		&& filter_width == weights.dim(2)
		&& filter_height == weights.dim(3));

	size_t n;
	n = fread((void *)weights.get_data(), sizeof(float), weights.num_elements(), f);
	if (n != weights.num_elements() || ferror(f))
		die(("fread weights: " + filename + " n=" + std::to_string(n)).c_str());

	n = fread((void *)bias.get_data(), sizeof(float), bias.num_elements(), f);
	if (n != bias.num_elements() || ferror(f))
		die(("fread bias: " + filename + " n=" + std::to_string(n)).c_str());

	if (fclose(f) != 0)
		die(filename.c_str());

	weights.moveToDevice();
	bias.moveToDevice();
}


