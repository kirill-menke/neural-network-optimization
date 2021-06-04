#include <assert.h>
#include <limits>

#include "MaxPool.h"

std::shared_ptr<Eigen::Tensor<float, 4>>
MaxPool::forward(std::shared_ptr<Eigen::Tensor<float, 4>> input_tensor)
{
	auto dims = input_tensor->dimensions();
	int batches = dims[0], channels = dims[1], width = dims[2], height = dims[3];
	assert(width == inputWidth && height == inputHeight);
	int outputWidth = this->outputWidth(), outputHeight = this->outputHeight();

	maximas = std::make_shared<Eigen::Tensor<int, 5>>(
		batches, channels, outputWidth, outputHeight, 2);

	auto output_tensor = std::make_shared<Eigen::Tensor<float, 4>>(
		batches, channels, outputWidth, outputHeight);

	for (int b = 0; b < batches; b++) {
		for (int c = 0; c < channels; c++) {
			for (int x_in = 0, x_out = 0; x_out < outputWidth; x_out++, x_in += strideX) {
				for (int y_in = 0, y_out = 0; y_out < outputHeight; y_out++, y_in += strideY) {
					int i_max = 0, j_max = 0, max = -std::numeric_limits<float>::infinity();

					for (int i = 0; i < poolWidth; i++) {
						for (int j = 0; j < poolHeight; j++) {
							float val = (*input_tensor)(b, c, x_in + i, y_in + j);
							if (val > max) {
								max = val;
								i_max = i;
								j_max = j;
							}
						}
					}

					(*maximas)(b, c, x_out, y_out, 0) = i_max;
					(*maximas)(b, c, x_out, y_out, 1) = j_max;
					(*output_tensor)(b, c, x_out, y_out) = max;
				}
			}
		}
	}

	return output_tensor;
}

std::shared_ptr<Eigen::Tensor<float, 4>>
MaxPool::backward(std::shared_ptr<Eigen::Tensor<float, 4>> error_tensor)
{
	auto dims = error_tensor->dimensions();
	int batches = dims[0], channels = dims[1], outputWidth = dims[2], outputHeight = dims[3];
	assert(outputWidth == this->outputWidth() && outputHeight == this->outputHeight());

	auto new_error = std::make_shared<Eigen::Tensor<float, 4>>(
		batches, channels, inputWidth, inputHeight);

	new_error->setZero();

	for (int b = 0; b < batches; b++) {
		for (int c = 0; c < channels; c++) {
			for (int x = 0; x < outputWidth; x++) {
				for (int y = 0; y < outputHeight; y++) {
					float err = (*error_tensor)(b, c, x, y);
					int i = (*maximas)(b, c, x, y, 0);
					int j = (*maximas)(b, c, x, y, 1);

					(*new_error)(b, c, x * strideX + i, y * strideY + j) += err;
				}
			}
		}
	}

	return new_error;
}
