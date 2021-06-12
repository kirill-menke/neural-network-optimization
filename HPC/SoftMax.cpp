#include <math.h>
#include <limits>

#include "SoftMax.h"
#include "Helper.h"

std::shared_ptr<Eigen::Tensor<float, 4>>
SoftMax::forward(std::shared_ptr<Eigen::Tensor<float, 4> const> input_tensor)
{
	auto dims = input_tensor->dimensions();
	int batches = dims[0], features = dims[1];

	output_tensor = std::make_shared<Eigen::Tensor<float, 4>>(batches, features, 1, 1);
	for (int b = 0; b < batches; b++) {
		float max = -std::numeric_limits<float>::infinity();
		for (int i = 0; i < features; i++) {
			float val = (*input_tensor)(b, i, 0, 0);
			if (val > max)
				max = val;
		}

		float sum = 0.;
		for (int i = 0; i < features; i++) {
			float val = expf((*input_tensor)(b, i, 0, 0) - max);
			(*output_tensor)(b, i, 0, 0) = val;
			sum += val;
		}

		for (int i = 0; i < features; i++) {
			(*output_tensor)(b, i, 0, 0) /= sum;
		}
	}

	printTensor(*output_tensor);

	return output_tensor;
}

std::shared_ptr<Eigen::Tensor<float, 4>>
SoftMax::backward(std::shared_ptr<Eigen::Tensor<float, 4> const> error_tensor)
{
	auto dims = error_tensor->dimensions();
	int batches = dims[0], features = dims[1];

	auto new_error = std::make_shared<Eigen::Tensor<float, 4>>(batches, features, 1, 1);

	for (int b = 0; b < batches; b++) {
		float sum = 0.;
		for (int i = 0; i < features; i++) {
			sum += (*error_tensor)(b, i, 0, 0) * (*output_tensor)(b, i, 0, 0);
		}

		for (int i = 0; i < features; i++) {
			(*new_error)(b, i, 0, 0) = (*output_tensor)(b, i, 0, 0) * ((*error_tensor)(b, i, 0, 0) - sum);
		}
	}

	return new_error;
}
