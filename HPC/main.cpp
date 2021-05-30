#pragma once
#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Conv.h"
#include "Helper.h"

int main() {
	// Parse image data, construct network, train and test network ...
	int batch_size = 1;
	int channels = 3;
	int filter_size = 3;
	int stride = 1;
	int num_kernels = 1;

	Eigen::Tensor<float, 4> input_tensor(batch_size, channels, 5, 5);
	input_tensor.setConstant(1);

	printTensor(input_tensor);

	Conv conv(num_kernels, channels, filter_size, stride);
	auto output_tensor = conv.forward(input_tensor);
	printTensor(output_tensor);

}


