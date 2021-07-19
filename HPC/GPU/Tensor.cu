#include "Tensor.h"

__global__
static void mergeKernel(Tensor<float, 4> t0, Tensor<float, 4> t1, Tensor<float, 4> tensor) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = tensor.dim(0), t0channels = t0.dim(1), t1channels = t1.dim(1),
	    width = tensor.dim(2), height = tensor.dim(3);

	if (x >= width || y >= height || b >= batchSize)
		return;

	int c;
	for (c = 0; c < t0channels; c++)
		tensor(b, c, x, y) = t0(b, c, x, y);

	for (int ct1 = 0; ct1 < t1channels; ct1++, c++)
		tensor(b, c, x, y) = t1(b, ct1, x, y);
}

Tensor<float, 4> mergeAtChannelDim(Tensor<float, 4> &a, Tensor<float, 4> &b) {
	int batchSize = a.dim(0), aChannels = a.dim(1), bChannels = b.dim(1), width = a.dim(2), height = a.dim(3);
	assert(b.dim(0) == batchSize && b.dim(2) == width && b.dim(3) == height);

	Tensor<float, 4> tensor({ batchSize, aChannels + bChannels, width, height });

	dim3 gridDim = getGridDim(width, height, batchSize);
	dim3 blockDim = getBlockDim(width, height, batchSize);
	mergeKernel<<<gridDim, blockDim>>>(a, b, tensor);

	return tensor;
}

__global__
static void splitKernel(Tensor<float, 4> t0, Tensor<float, 4> t1, Tensor<float, 4> tensor) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = tensor.dim(0), t0channels = t0.dim(1), t1channels = t1.dim(1),
	    width = tensor.dim(2), height = tensor.dim(3);

	if (x >= width || y >= height || b >= batchSize)
		return;

	for (int c = 0; c < t0channels; c++)
		t0(b, c, x, y) = tensor(b, c, x, y);

	for (int c = 0; c < t1channels; c++)
		t1(b, c, x, y) = tensor(b, c + t0channels, x, y);
}

std::pair<Tensor<float, 4>, Tensor<float, 4>> splitAtChannelDim(Tensor<float, 4> &tensor, int channel) {
	int batchSize = tensor.dim(0), channels = tensor.dim(1), width = tensor.dim(2), height = tensor.dim(3);
	assert(0 < channel && channel < channels);

	Tensor<float, 4> t0({ batchSize, channel, width, height });
	Tensor<float, 4> t1({ batchSize, channels - channel, width, height });

	dim3 gridDim = getGridDim(width, height, batchSize);
	dim3 blockDim = getBlockDim(width, height, batchSize);
	splitKernel<<<gridDim, blockDim>>>(t0, t1, tensor);

	return std::pair<Tensor<float, 4>, Tensor<float, 4>>(t0, t1);
}


__global__
static void addTensorsKernel(Tensor<float, 4> res, Tensor<float, 4> t1, Tensor<float, 4> t2) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int b = threadIdx.z + blockIdx.z * blockDim.z;

	int batchSize = res.dim(0), channels = res.dim(1),
	    width = res.dim(2), height = res.dim(3);

	if (x >= width || y >= height || b >= batchSize)
		return;

	for (int c = 0; c < channels; c++)
		res(b, c, x, y) = t1(b, c, x, y) + t2(b, c, x, y);
}

Tensor<float, 4> operator+(Tensor<float, 4> &a, Tensor<float, 4> &b) {
	assert(a.dim(0) == b.dim(0));
	assert(a.dim(1) == b.dim(1));
	assert(a.dim(2) == b.dim(2));
	assert(a.dim(3) == b.dim(3));

	Tensor<float, 4> res({ a.dim(0), a.dim(1), a.dim(2), a.dim(3) });

	dim3 gridDim = getGridDim(a.dim(2), a.dim(3), a.dim(0));
	dim3 blockDim = getBlockDim(a.dim(2), a.dim(3), a.dim(0));
	addTensorsKernel<<<gridDim, blockDim>>>(res, a, b);

	return res;
}

