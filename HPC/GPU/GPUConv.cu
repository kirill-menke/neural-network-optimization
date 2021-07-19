#include <assert.h>

#include "./GPUConv.h"
#include "./cuda-utils.h"
#include "./conv-utils.h"



Tensor<float, 4> GPUConv::forward(const Tensor<float, 4> &input_tensor) {
	int batchSize = input_tensor.dim(0);

	assert(input_tensor.dim(1) == inputChannels);
	assert(input_tensor.dim(2) == imageWidth);
	assert(input_tensor.dim(3) == imageHeight);

	if (this->padded_input == nullptr)
		this->padded_input = new Tensor<float, 4>({
			batchSize, inputChannels,
			imageWidth + 2 * (filterWidth / 2),
			imageHeight + 2 * (filterHeight / 2)
		});

	{
		dim3 gridDim = getGridDim(padded_input->dim(2), padded_input->dim(3), batchSize);
		dim3 blockDim = getBlockDim(padded_input->dim(2), padded_input->dim(3), batchSize);
		addPadding<<<gridDim, blockDim>>>(input_tensor, *padded_input, filterWidth / 2, filterHeight / 2);
	}

	int outputWidth = imageWidth / strideX, outputHeight = imageHeight / strideY;

	Tensor<float, 4> output_tensor({
		batchSize,
		outputChannels,
		outputWidth,
		outputHeight
	});

	{
		dim3 gridDim = getGridDim(outputWidth, outputHeight, batchSize);
		dim3 blockDim = getBlockDim(outputWidth, outputHeight, batchSize);
		convolution<false><<<gridDim, blockDim>>>(*padded_input, output_tensor, weights, bias, strideX, strideY);
	}

	return output_tensor;
}

/*
 * Opt./Simplify for stride = 1?
 */
Tensor<float, 4> GPUConv::backward(const Tensor<float, 4> &error_tensor) {
	int batchSize = error_tensor.dim(0);
	int outputWidth = imageWidth / strideX, outputHeight = imageHeight / strideY;
	assert(error_tensor.dim(1) == outputChannels);
	assert(error_tensor.dim(2) == outputWidth);
	assert(error_tensor.dim(3) == outputHeight);
	assert(padded_input->dim(0) == batchSize);

#if 1
	assert(strideX == 1 && strideY == 1);
	Tensor<float, 4> upsampled_error_tensor(error_tensor);
#else
	Tensor<float, 4> upsampled_error_tensor({
		batchSize,
		outputChannels,
		imageWidth,
		imageHeight
	});

	{
		dim3 gridDim = getGridDim(imageWidth, imageHeight, batchSize);
		dim3 blockDim = getBlockDim(imageWidth, imageHeight, batchSize);
		upsample<<<gridDim, blockDim>>>(error_tensor, upsampled_error_tensor, strideX, strideY);
	}
#endif

	Tensor<float, 4> padded_error_tensor({
		batchSize, outputChannels,
		imageWidth + 2 * (filterWidth / 2),
		imageHeight + 2 * (filterHeight / 2)
	});

	{
		dim3 gridDim = getGridDim(padded_error_tensor.dim(2), padded_error_tensor.dim(3), batchSize);
		dim3 blockDim = getBlockDim(padded_error_tensor.dim(2), padded_error_tensor.dim(3), batchSize);
		addPadding<<<gridDim, blockDim>>>(upsampled_error_tensor, padded_error_tensor, filterWidth / 2, filterHeight / 2);
	}


	Tensor<float, 4> next_error_tensor({
		batchSize, inputChannels,
		imageWidth, imageHeight
	});

	{
		dim3 gridDim = getGridDim(imageWidth, imageHeight, batchSize);
		dim3 blockDim = getBlockDim(imageWidth, imageHeight, batchSize);
		convolution<true><<<gridDim, blockDim>>>(padded_error_tensor, next_error_tensor, weights, bias, 1, 1);
	}

	Tensor<float, 4> gradient_weights({
		outputChannels, inputChannels,
		filterWidth, filterHeight });

	Tensor<float, 1> gradient_bias({ outputChannels });

	{
		dim3 gridDim = getGridDim(filterWidth, filterHeight, outputChannels);
		dim3 blockDim = getBlockDim(filterWidth, filterHeight, outputChannels);
		backwardGradients<<<gridDim, blockDim>>>(error_tensor, upsampled_error_tensor,
				*padded_input, gradient_weights, gradient_bias, strideX, strideY);
	}

	if (this->optimizer != nullptr)
		this->optimizer->update(weights, bias, gradient_weights, gradient_bias);

	return next_error_tensor;
}
