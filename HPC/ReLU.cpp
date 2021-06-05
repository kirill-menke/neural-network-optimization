
#include "ReLU.h"

std::shared_ptr<Eigen::Tensor<float, 4>>
ReLU::forward(std::shared_ptr<Eigen::Tensor<float, 4>> input_tensor)
{
	auto dims = input_tensor->dimensions();
	int batches = dims[0], channels = dims[1], width = dims[2], height = dims[3];

	output_tensor = std::make_shared<Eigen::Tensor<float, 4>>(batches, channels, width, height);

	for (int b = 0; b < batches; b++) {
		for (int c = 0; c < channels; c++) {
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					float val = (*input_tensor)(b, c, x, y);
					(*output_tensor)(b, c, x, y) = val < 0.
						? 0.
						: val;
				}
			}
		}
	}

	return output_tensor;
}

std::shared_ptr<Eigen::Tensor<float, 4>>
ReLU::backward(std::shared_ptr<Eigen::Tensor<float, 4>> error_tensor)
{
	auto dims = error_tensor->dimensions();
	int batches = dims[0], channels = dims[1], width = dims[2], height = dims[3];

	auto new_error = std::make_shared<Eigen::Tensor<float, 4>>(batches, channels, width, height);
	for (int b = 0; b < batches; b++) {
		for (int c = 0; c < channels; c++) {
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					(*new_error)(b, c, x, y) = (*output_tensor)(b, c, x, y) < 0.
						? 0.
						: (*error_tensor)(b, c, x, y);
				}
			}
		}
	}

	return new_error;
}
