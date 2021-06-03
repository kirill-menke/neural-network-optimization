#include <stdlib.h>
#include <stdio.h>

#include "./GPUConv.h"

int main() {
	printf("Hello World!\n");

	int batchSize = 1;
	int inputChannels = 1;
	int outputChannels = 1;
	int filterWidth = 3;
	int filterHeight = 3;
	int imageWidth = 5;
	int imageHeight = 5;

	GPUConv convLayer(
		inputChannels,
		imageWidth,
		imageHeight,
		outputChannels,
		filterWidth,
		filterHeight);

	convLayer.filters.moveToHost();
	convLayer.filters(0, 0, 0, 0) = 1.;
	convLayer.filters(0, 0, 1, 0) = 1.;
	convLayer.filters(0, 0, 2, 0) = 1.;
	convLayer.filters(0, 0, 0, 1) = 1.;
	convLayer.filters(0, 0, 1, 1) = 1.;
	convLayer.filters(0, 0, 2, 1) = 1.;
	convLayer.filters(0, 0, 0, 2) = 1.;
	convLayer.filters(0, 0, 1, 2) = 1.;
	convLayer.filters(0, 0, 2, 2) = 1.;
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


	return EXIT_SUCCESS;
}

