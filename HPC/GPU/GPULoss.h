#include "./Tensor.h"


class GPUCrossEntropyLoss {
public:

	GPUCrossEntropyLoss(int imageWidth, int imageHeight) : 
		imageWidth(imageWidth), imageHeight(imageHeight) {}

	/* The label_tensor should be one-hot-encoded! */
	Tensor<float, 2> forward(Tensor<float, 4>& pred, Tensor<float, 3>& target);

	Tensor<float, 4> backward(Tensor<float, 4>& pred, Tensor<float, 3>& target);

private:
	int imageWidth, imageHeight;
};
