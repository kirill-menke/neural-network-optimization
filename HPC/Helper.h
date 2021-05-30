#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

static void printTensor(Eigen::Tensor<float, 4> tensor) {
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