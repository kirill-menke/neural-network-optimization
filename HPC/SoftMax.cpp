#include <math.h>
#include <limits>

#include "SoftMax.h"

std::shared_ptr<Eigen::Tensor<float, 2>>
SoftMax::forward(std::shared_ptr<Eigen::Tensor<float, 2>> input_tensor)
{
	auto dims = input_tensor->dimensions();
	int batches = dims[0], features = dims[1];

	output_tensor = std::make_shared<Eigen::Tensor<float, 2>>(batches, features);
	for (int b = 0; b < batches; b++) {
		float max = -std::numeric_limits<float>::infinity();
		for (int i = 0; i < features; i++) {
			float val = (*input_tensor)(b, i);
			if (val > max)
				max = val;
		}

		float sum = 0.;
		for (int i = 0; i < features; i++) {
			float val = expf((*input_tensor)(b, i) - max);
			(*output_tensor)(b, i) = val;
			sum += val;
		}

		for (int i = 0; i < features; i++) {
			(*output_tensor)(b, i) /= sum;
		}
	}

	return output_tensor;
}

std::shared_ptr<Eigen::Tensor<float, 2>>
SoftMax::backward(std::shared_ptr<Eigen::Tensor<float, 2>> error_tensor)
{
	auto dims = error_tensor->dimensions();
	int batches = dims[0], features = dims[1];

	auto new_error = std::make_shared<Eigen::Tensor<float, 2>>(batches, features);

	for (int b = 0; b < batches; b++) {
		float sum = 0.;
		for (int i = 0; i < features; i++) {
			sum += (*error_tensor)(b, i) * (*output_tensor)(b, i);
		}

		for (int i = 0; i < features; i++) {
			(*new_error)(b, i) = (*output_tensor)(b, i) * ((*error_tensor)(b, i) - sum);
		}
	}

	return new_error;
}
