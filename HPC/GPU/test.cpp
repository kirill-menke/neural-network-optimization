#include <stdlib.h>
#include <stdio.h>

#include "./GPUConv.h"

int main() {
	printf("Hello World!\n");

#if 0	
	int batchSize = 1;
	int inputChannels = 1;
	int outputChannels = 1;
	int filterWidth = 3;
	int filterHeight = 3;
	int imageWidth = 5;
	int imageHeight = 5;
	int strideX = 1;
	int strideY = 1;

	GPUConv convLayer(
		inputChannels,
		imageWidth,
		imageHeight,
		outputChannels,
		filterWidth,
		filterHeight,
		strideX,
		strideY);

	convLayer.filters.moveToHost();
	convLayer.filters(0, 0, 0, 0) = 0.;
	convLayer.filters(0, 0, 1, 0) = 1.;
	convLayer.filters(0, 0, 2, 0) = 2.;
	convLayer.filters(0, 0, 0, 1) = 0.;
	convLayer.filters(0, 0, 1, 1) = 1.;
	convLayer.filters(0, 0, 2, 1) = 2.;
	convLayer.filters(0, 0, 0, 2) = 0.;
	convLayer.filters(0, 0, 1, 2) = 1.;
	convLayer.filters(0, 0, 2, 2) = 2.;
	convLayer.filters.dump(stdout, "Filter");
	convLayer.filters.moveToDevice();

	Tensor<float, 4> imageIn(Tensor<float, 4>::ON_CPU, { batchSize, inputChannels, imageWidth, imageHeight });
	imageIn(0, 0, 0, 0) = 0.;
	imageIn(0, 0, 1, 0) = 1.;
	imageIn(0, 0, 2, 0) = 2.;
	imageIn(0, 0, 3, 0) = 3.;
	imageIn(0, 0, 4, 0) = 4.;
	imageIn(0, 0, 0, 1) = 5.;
	imageIn(0, 0, 1, 1) = 6.;
	imageIn(0, 0, 2, 1) = 7.;
	imageIn(0, 0, 3, 1) = 8.;
	imageIn(0, 0, 4, 1) = 9.;
	imageIn(0, 0, 0, 2) = 10.;
	imageIn(0, 0, 1, 2) = 11.;
	imageIn(0, 0, 2, 2) = 12.;
	imageIn(0, 0, 3, 2) = 13.;
	imageIn(0, 0, 4, 2) = 14.;
	imageIn(0, 0, 0, 3) = 15.;
	imageIn(0, 0, 1, 3) = 16.;
	imageIn(0, 0, 2, 3) = 17.;
	imageIn(0, 0, 3, 3) = 18.;
	imageIn(0, 0, 4, 3) = 19.;
	imageIn(0, 0, 0, 4) = 20.;
	imageIn(0, 0, 1, 4) = 21.;
	imageIn(0, 0, 2, 4) = 22.;
	imageIn(0, 0, 3, 4) = 23.;
	imageIn(0, 0, 4, 4) = 24.;
	imageIn.dump(stdout, "Input Image");
	imageIn.moveToDevice();

	Tensor<float, 4> imageOut = convLayer.forward(imageIn);
	imageOut.moveToHost();
	imageOut.dump(stdout, "Output Image");

	Tensor<float, 4> error_tensor(Tensor<float, 4>::ON_CPU, { batchSize, outputChannels, imageWidth, imageHeight });
	error_tensor(0, 0, 0, 0) = 0.;
	error_tensor(0, 0, 1, 0) = 1.;
	error_tensor(0, 0, 2, 0) = 2.;
	error_tensor(0, 0, 3, 0) = 3.;
	error_tensor(0, 0, 4, 0) = 4.;
	error_tensor(0, 0, 0, 1) = 5.;
	error_tensor(0, 0, 1, 1) = 6.;
	error_tensor(0, 0, 2, 1) = 7.;
	error_tensor(0, 0, 3, 1) = 8.;
	error_tensor(0, 0, 4, 1) = 9.;
	error_tensor(0, 0, 0, 2) = 10.;
	error_tensor(0, 0, 1, 2) = 11.;
	error_tensor(0, 0, 2, 2) = 12.;
	error_tensor(0, 0, 3, 2) = 13.;
	error_tensor(0, 0, 4, 2) = 14.;
	error_tensor(0, 0, 0, 3) = 15.;
	error_tensor(0, 0, 1, 3) = 16.;
	error_tensor(0, 0, 2, 3) = 17.;
	error_tensor(0, 0, 3, 3) = 18.;
	error_tensor(0, 0, 4, 3) = 19.;
	error_tensor(0, 0, 0, 4) = 20.;
	error_tensor(0, 0, 1, 4) = 21.;
	error_tensor(0, 0, 2, 4) = 22.;
	error_tensor(0, 0, 3, 4) = 23.;
	error_tensor(0, 0, 4, 4) = 24.;
	error_tensor.dump(stdout, "Error Tensor");
	error_tensor.moveToDevice();

	Tensor<float, 4> next_error_tensor = convLayer.backward(error_tensor);
	next_error_tensor.moveToHost();
	next_error_tensor.dump(stdout, "Next Error Tensor");
#endif


	int batchSize = 1;
	int inputChannels = 1;
	int outputChannels = 1;
	int filterWidth = 3;
	int filterHeight = 3;
	int imageWidth = 5;
	int imageHeight = 5;
	int strideX = 2;
	int strideY = 2;

	GPUConv convLayer(
		inputChannels,
		imageWidth,
		imageHeight,
		outputChannels,
		filterWidth,
		filterHeight,
		strideX,
		strideY);

	convLayer.filters.moveToHost();
	convLayer.filters(0, 0, 0, 0) = 0.;
	convLayer.filters(0, 0, 1, 0) = 1.;
	convLayer.filters(0, 0, 2, 0) = 2.;
	convLayer.filters(0, 0, 0, 1) = 0.;
	convLayer.filters(0, 0, 1, 1) = 1.;
	convLayer.filters(0, 0, 2, 1) = 2.;
	convLayer.filters(0, 0, 0, 2) = 0.;
	convLayer.filters(0, 0, 1, 2) = 1.;
	convLayer.filters(0, 0, 2, 2) = 2.;
	convLayer.filters.dump(stdout, "Filter");
	convLayer.filters.moveToDevice();

	Tensor<float, 4> imageIn(Tensor<float, 4>::ON_CPU, { batchSize, inputChannels, imageWidth, imageHeight });
	imageIn(0, 0, 0, 0) = 0.;
	imageIn(0, 0, 1, 0) = 1.;
	imageIn(0, 0, 2, 0) = 2.;
	imageIn(0, 0, 3, 0) = 3.;
	imageIn(0, 0, 4, 0) = 4.;
	imageIn(0, 0, 0, 1) = 5.;
	imageIn(0, 0, 1, 1) = 6.;
	imageIn(0, 0, 2, 1) = 7.;
	imageIn(0, 0, 3, 1) = 8.;
	imageIn(0, 0, 4, 1) = 9.;
	imageIn(0, 0, 0, 2) = 10.;
	imageIn(0, 0, 1, 2) = 11.;
	imageIn(0, 0, 2, 2) = 12.;
	imageIn(0, 0, 3, 2) = 13.;
	imageIn(0, 0, 4, 2) = 14.;
	imageIn(0, 0, 0, 3) = 15.;
	imageIn(0, 0, 1, 3) = 16.;
	imageIn(0, 0, 2, 3) = 17.;
	imageIn(0, 0, 3, 3) = 18.;
	imageIn(0, 0, 4, 3) = 19.;
	imageIn(0, 0, 0, 4) = 20.;
	imageIn(0, 0, 1, 4) = 21.;
	imageIn(0, 0, 2, 4) = 22.;
	imageIn(0, 0, 3, 4) = 23.;
	imageIn(0, 0, 4, 4) = 24.;
	imageIn.dump(stdout, "Input Image");
	imageIn.moveToDevice();

	Tensor<float, 4> imageOut = convLayer.forward(imageIn);
	imageOut.moveToHost();
	imageOut.dump(stdout, "Output Image");

	Tensor<float, 4> error_tensor(Tensor<float, 4>::ON_CPU, { batchSize, outputChannels, imageWidth / strideX, imageHeight / strideY });
	error_tensor(0, 0, 0, 0) = 0.;
	error_tensor(0, 0, 1, 0) = 1.;
	error_tensor(0, 0, 0, 1) = 2.;
	error_tensor(0, 0, 1, 1) = 3.;
	error_tensor.dump(stdout, "Error Tensor");
	error_tensor.moveToDevice();

	Tensor<float, 4> next_error_tensor = convLayer.backward(error_tensor);
	next_error_tensor.moveToHost();
	next_error_tensor.dump(stdout, "Next Error Tensor");

	return EXIT_SUCCESS;
}

