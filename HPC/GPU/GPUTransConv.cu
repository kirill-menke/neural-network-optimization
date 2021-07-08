#include "GPUTransConv.h"
#include "conv-utils.h"


Tensor<float, 4> GPUTransConv::forward(Tensor<float, 4>& input_tensor) {
	int batchSize = input_tensor.dim(0);
	int upsampledWidth = imageWidth * strideX - (strideX - 1),
		upsampledHeight = imageHeight * strideY - (strideY - 1),
		paddingWidth = 2 * (filterWidth - 1),
		paddingHeight = 2 * (filterHeight - 1),
		outputWidth = strideX * (imageWidth - 1) + weights.dim(2),
		outputHeight = strideY * (imageHeight - 1) + weights.dim(3);


	Tensor<float, 4> upsampled_tensor({ batchSize, outputChannels, upsampledWidth, upsampledHeight });
	{
		dim3 gridDim = getGridDim(imageWidth, imageHeight, batchSize);
		dim3 blockDim = getBlockDim(imageWidth, imageHeight, batchSize);
		upsample<<<gridDim, blockDim>>> (input_tensor, upsampled_tensor, strideX, strideY);
	}


	if (this->padded_input == nullptr) {
		this->padded_input = new Tensor<float, 4>({ batchSize, outputChannels, upsampledWidth + paddingWidth, upsampledHeight + paddingHeight });
	}
	{
		dim3 gridDim = getGridDim(padded_input->dim(2), padded_input->dim(3), batchSize);
		dim3 blockDim = getBlockDim(padded_input->dim(2), padded_input->dim(3), batchSize);
		addPadding<<<gridDim, blockDim>>> (upsampled_tensor, *padded_input, filterWidth - 1, filterHeight - 1);
	}


	Tensor<float, 4> output_tensor({ batchSize, inputChannels, outputWidth, outputHeight});	
	{
		dim3 gridDim = getGridDim(upsampledWidth + 1, upsampledHeight, batchSize);
		dim3 blockDim = getBlockDim(upsampledWidth + 1, upsampledHeight, batchSize);
		convolution<false> <<<gridDim, blockDim>>> (*padded_input, output_tensor, weights, bias, 1, 1);
	}

	return output_tensor;
}


Tensor<float, 4> GPUTransConv::backward(Tensor<float, 4>& error_tensor) {
	int batchSize = error_tensor.dim(0);

	Tensor<float, 4> next_error_tensor({ batchSize, outputChannels, imageWidth, imageHeight});
	{
		dim3 gridDim = getGridDim(imageWidth, imageHeight, batchSize);
		dim3 blockDim = getBlockDim(imageWidth, imageHeight, batchSize);
		convolution<true> <<<gridDim, blockDim>>> (error_tensor, next_error_tensor, weights, bias, strideX, strideY);
	}


	Tensor<float, 4> gradient_weights({ outputChannels, inputChannels, filterWidth, filterHeight });
	Tensor<float, 1> gradient_bias({ outputChannels });
	{
		dim3 gridDim = getGridDim(filterWidth, filterHeight, outputChannels);
		dim3 blockDim = getBlockDim(filterWidth, filterHeight, outputChannels);
		backwardGradients<<<gridDim, blockDim>>> (error_tensor, error_tensor, *padded_input, gradient_weights, gradient_bias, strideX, strideY);
	}


	if (this->optimizer != nullptr)
		this->optimizer->update(weights, bias, gradient_weights, gradient_bias);


	return next_error_tensor;
}