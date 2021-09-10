#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

class Optimizer;
class Initializer;

class Conv {
	int stride;
	int num_kernels;
	int channels;
	int filter_size;

	Optimizer* optimizer;
	Eigen::Tensor<float, 4> weights;
	Eigen::Tensor<float, 1> bias;
	std::shared_ptr<Eigen::Tensor<float, 4>> input_tensor;
	std::shared_ptr<Eigen::Tensor<float, 1>> gradient_bias;
	std::shared_ptr<Eigen::Tensor<float, 4>> gradient_weights;
	

public:
	Conv(int num_kernels, int channels, int filter_size, int stride);
	
	/** Forward pass: Convolves input_tensor with internal weights */
	std::shared_ptr<Eigen::Tensor<float, 4>> forward(std::shared_ptr<Eigen::Tensor<float, 4>> input_tensor);

	/** Backward pass: Computes gradient gradient w.r.t. the input (backpropagation), 
	and gradient w.r.t. the weights which is used to update the weights (gradient descent) */
	std::shared_ptr<Eigen::Tensor<float, 4>> backward(std::shared_ptr<Eigen::Tensor<float, 4>> error_tensor);

	void setOptimizer(Optimizer* optimizer);
	void setInitializer(Initializer* initializer);
	void setWeights(Eigen::Tensor<float, 4> weights);
	void setBias(Eigen::Tensor<float, 1> bias);

	Eigen::Tensor<float, 4> getWeights();
	Eigen::Tensor<float, 1> getBias();
	Eigen::Tensor<float, 4> getGradientWeights();
	Eigen::Tensor<float, 1> getGradientBias();
	std::array<int, 4> getWeightDims();


private:
	/** Pads the border of the inputs spatial dimension with zeros */
	std::shared_ptr<Eigen::Tensor<float, 4>> pad(std::shared_ptr<Eigen::Tensor<float, 4>> input, int px, int py);
	
	/** Upsamples the input using the stride: Returns tensor with the original input size */
	std::shared_ptr<Eigen::Tensor<float, 4>> upsample(std::shared_ptr<Eigen::Tensor<float, 4>> input);

	std::shared_ptr<Eigen::Tensor<float, 4>> convolutionForward(std::shared_ptr<Eigen::Tensor<float, 4>> input, int output_width, int output_height);
	std::shared_ptr<Eigen::Tensor<float, 4>> convolutionBackward(std::shared_ptr<Eigen::Tensor<float, 4>> input, int output_width, int output_height);
};
