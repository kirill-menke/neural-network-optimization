#include <vector>
#include <unordered_map>

#include "tensor.h"

static std::unordered_map<size_t, std::vector<void*>> prev_allocs;

void *alloc_gpu_memory(size_t bytes) {
	void *data;
	std::vector<void*> &pointers = prev_allocs[bytes];
	if (pointers.size() > 0) {
		data = pointers.back();
		pointers.pop_back();
		return data;
	}

	cudaErrchk(cudaMalloc((void **)&data, bytes));
	return data;
}

void free_gpu_memory(void *ptr, size_t bytes) {
	// cudaErrchk(cudaFree(ptr));
	prev_allocs[bytes].push_back(ptr);
}

__global__
static void concat_kernel(Tensor<float, 4> t0, Tensor<float, 4> t1, Tensor<float, 4> tensor) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = tensor.dim(0), t0channels = t0.dim(1), t1channels = t1.dim(1),
	    width = tensor.dim(2), height = tensor.dim(3);

	if (x >= width || y >= height || c >= t0channels + t1channels)
		return;

	for (int b = 0; b < batch_size; b++) {
		if (c < t0channels)
			tensor(b, c, x, y) = t0(b, c, x, y);
		else
			tensor(b, c, x, y) = t1(b, c - t0channels, x, y);
	}
}

Tensor<float, 4> concat(const Tensor<float, 4> &a, const Tensor<float, 4> &b) {
	int batch_size = a.dim(0), a_channels = a.dim(1), b_channels = b.dim(1),
	    width = a.dim(2), height = a.dim(3);
	assert(b.dim(0) == batch_size);
	assert(b.dim(2) == width && b.dim(3) == height);

	Tensor<float, 4> tensor(batch_size, a_channels + b_channels, width, height);

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, a_channels + b_channels, width, height);
	concat_kernel<<<gridDim, blockDim>>>(a, b, tensor);

	return tensor;
}

__global__
static void split_kernel(Tensor<float, 4> t0, Tensor<float, 4> t1, Tensor<float, 4> tensor) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = tensor.dim(0),
	    channels = tensor.dim(1),
	    t0channels = t0.dim(1),
	    t1channels = t1.dim(1),
	    width = tensor.dim(2), height = tensor.dim(3);

	if (x >= width || y >= height || c >= channels)
		return;

	for (int b = 0; b < batch_size; b++) {
		if (c < t0channels)
			t0(b, c, x, y) = tensor(b, c, x, y);
		else
			t1(b, c - t0channels, x, y) = tensor(b, c, x, y);
	}
}

std::pair<Tensor<float, 4>, Tensor<float, 4>> split(const Tensor<float, 4> &tensor, int channel) {
	int batchSize = tensor.dim(0), channels = tensor.dim(1),
	    width = tensor.dim(2), height = tensor.dim(3);
	assert(0 < channel && channel < channels);

	Tensor<float, 4> t0(batchSize, channel, width, height);
	Tensor<float, 4> t1(batchSize, channels - channel, width, height);

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, channels, width, height);
	split_kernel<<<gridDim, blockDim>>>(t0, t1, tensor);

	return std::pair<Tensor<float, 4>, Tensor<float, 4>>(t0, t1);
}

__global__
static void add_tensors_kernel(Tensor<float, 4> res, Tensor<float, 4> t1, Tensor<float, 4> t2) {
	int y = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z + blockIdx.z * blockDim.z;

	int batch_size = res.dim(0), channels = res.dim(1),
	    width = res.dim(2), height = res.dim(3);

	if (x >= width || y >= height || c >= channels)
		return;

	for (int b = 0; b < batch_size; b++)
		res(b, c, x, y) = t1(b, c, x, y) + t2(b, c, x, y);
}

Tensor<float, 4> operator+(const Tensor<float, 4> &a, const Tensor<float, 4> &b) {
	assert(a.dim(0) == b.dim(0));
	assert(a.dim(1) == b.dim(1));
	assert(a.dim(2) == b.dim(2));
	assert(a.dim(3) == b.dim(3));

	Tensor<float, 4> res(a.dim(0), a.dim(1), a.dim(2), a.dim(3));

	dim3 gridDim;
	dim3 blockDim;
	getGridSize(gridDim, blockDim, a.dim(1), a.dim(2), a.dim(3));
	add_tensors_kernel<<<gridDim, blockDim>>>(res, a, b);

	return res;
}

