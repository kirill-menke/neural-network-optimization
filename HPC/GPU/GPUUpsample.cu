#include "GPUUpsample.h"
#include "conv-utils.h"
#include "cuda-utils.h"
#include <cstdio>

__global__
static void upsampling_error(Tensor<float, 4> error_tensor, Tensor<float, 4> next_error_tensor, int strideX, int strideY) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = error_tensor.dim(0), channels = error_tensor.dim(1);
	int output_width = next_error_tensor.dim(2), output_height = next_error_tensor.dim(3);

	if (x >= output_width || y >= output_height || b >= batch_size)
		return;

	for (int c = 0; c < channels; c++) {
		float val = 0.;
		for (int w = 0; w < strideX; w++) {
			for (int h = 0; h < strideY; h++) {
				val += error_tensor(b, c, x * strideX + w, y * strideY + h);
			}
		}

		next_error_tensor(b, c, x, y) = val;
	}
}


Tensor<float, 4> GPUUpsample::forward(Tensor<float, 4>& input_tensor) {
	int batchSize = input_tensor.dim(0),
		channels = input_tensor.dim(1);
	int upsampledWidth = imageWidth * strideX,
		upsampledHeight = imageHeight * strideY;


	Tensor<float, 4> upsampled_tensor({ batchSize, channels, upsampledWidth, upsampledHeight });
	{
		dim3 gridDim = getGridDim(upsampledWidth, upsampledHeight, batchSize);
		dim3 blockDim = getBlockDim(upsampledWidth, upsampledHeight, batchSize);
		upsample_neighbor<<<gridDim, blockDim>>> (input_tensor, upsampled_tensor, strideX, strideY);
	}

	return upsampled_tensor;
}


Tensor<float, 4> GPUUpsample::backward(Tensor<float, 4>& error_tensor) {
	int batchSize = error_tensor.dim(0),
		channels = error_tensor.dim(1),
		width = error_tensor.dim(2),
		height = error_tensor.dim(3);

	Tensor<float, 4> next_error_tensor({ batchSize, channels, imageWidth, imageHeight });
	{
		dim3 gridDim = getGridDim(imageWidth, imageHeight, batchSize);
		dim3 blockDim = getBlockDim(imageWidth, imageHeight, batchSize);
		upsampling_error<<<gridDim, blockDim>>> (error_tensor, next_error_tensor, strideX, strideY);
	}

	return next_error_tensor;
}
