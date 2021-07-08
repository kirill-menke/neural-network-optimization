#pragma once
#include "Tensor.h"
#include "GPUOptimizer.h"

class GPUUpsample{
public:
	GPUUpsample(
		int imageWidth,
		int imageHeight,
		int strideX,
		int strideY
	) :
		imageWidth(imageWidth),
		imageHeight(imageHeight),
		strideX(strideX),
		strideY(strideY) {}

	virtual Tensor<float, 4> forward(Tensor<float, 4>& input_tensor);
	virtual Tensor<float, 4> backward(Tensor<float, 4>& error_tensor);

	int imageWidth;
	int imageHeight;
	int strideX;
	int strideY;

};