#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "utils.h"

#ifdef __linux__
// See here: https://stackoverflow.com/a/43183942/5682784
namespace std {
	static inline float sqrtf(float x) {
		return ::sqrt(x);
	}
}
#else
#endif

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

void heInit(Tensor<float, 4>& weights, Tensor<float, 1>& bias) {
	float std_weights = std::sqrtf(2 / static_cast<float>(weights.dim(1) * weights.dim(2) * weights.dim(3)));
	float std_bias = std::sqrtf(2 / static_cast<float>(weights.dim(0)));
	std::normal_distribution<float> distribution_weights(0.0, std_weights);
	std::normal_distribution<float> distribution_bias(0.0, std_bias);

	for (int i = 0; i < weights.dim(0); i++) {
		for (int j = 0; j < weights.dim(1); j++) {
			for (int x = 0; x < weights.dim(2); x++) {
				for (int y = 0; y < weights.dim(3); y++) {
					weights(i, j, x, y) = distribution_weights(rng);
				}
			}
		}
	}

	for (int i = 0; i < bias.dim(0); i++) {
		bias(i) = distribution_bias(rng);
	}
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

	FILE* f = fopen(filename.c_str(), "w");
	if (f == NULL)
		die(filename.c_str());

	size_t n;
	fprintf(f, "%d %d %d %d:", output_channels, input_channels, filter_width, filter_height);
	n = fwrite((void *)weights.get_data(), sizeof(float), weights.num_elements(), f);
	if (n != weights.num_elements())
		die(filename.c_str());

	n = fwrite((void *)bias.get_data(), sizeof(float), bias.num_elements(), f);
	if (n != bias.num_elements())
		die(filename.c_str());

	if (ferror(f))
		die(filename.c_str());
	if (fclose(f) != 0)
		die(filename.c_str());
}

void readFromFile(const std::string &filename, Tensor<float, 4> &weights, Tensor<float, 1> &bias) {
	assert(bias.dim(0) == weights.dim(0));
	int output_channels, input_channels, filter_width, filter_height;

	FILE* f = fopen(filename.c_str(), "r");
	if (f == NULL)
		die(filename.c_str());

	if (fscanf(f, "%d %d %d %d:", &output_channels, &input_channels, &filter_width, &filter_height) != 4)
		die(filename.c_str());

	assert(output_channels == weights.dim(0)
		&& input_channels == weights.dim(1)
		&& filter_width == weights.dim(2)
		&& filter_height == weights.dim(3));

	size_t n,
	       weights_elms = weights.num_elements(),
	       bias_elms = bias.num_elements();

	n = fread((void *)weights.get_data(), sizeof(float), weights_elms, f);
	if (n != weights_elms)
		die(filename.c_str());

	n = fread((void *)bias.get_data(), sizeof(float), bias_elms, f);
	if (n != bias_elms)
		die(filename.c_str());

	if (ferror(f) || fclose(f) != 0)
		die(filename.c_str());

	weights.moveToDevice();
	bias.moveToDevice();
}

