#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include "Layer.h"

class FullyConnected : Layer {
	int input_size;
	int output_size;

	Eigen::MatrixXf weights;
	
public:
	FullyConnected(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
		trainable = true;

		weights = Eigen::MatrixXf(input_size + 1, output_size);
	}


}