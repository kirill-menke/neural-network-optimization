#pragma once

#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <initializer_list>
#include <utility>

#include "cuda-utils.h"

template<typename Scalar, size_t Rank>
class Tensor {
public:
	/*
	 * This class copies by reference! It does NOT behave like other
	 * C++ containers or Eigen::Tensor. It does reference counting internally,
	 * so explicitly calling `destroy` is not needed.
	 *
	 * There is currently no way of doing a "deep copy".
	 */
	Tensor(std::initializer_list<int> dims, bool do_allocs = true) {
		assert(dims.size() == Rank);
		auto iter = dims.begin();
		for (int i = 0; i < Rank; i++, iter++)
			this->dims[i] = *iter;

		if (do_allocs) {
			this->refcount = new int(1);
			this->data = (Scalar *)calloc(this->num_elements(), sizeof(Scalar));
			cudaErrchk(cudaMalloc(
				(void **)&this->dev_data,
				this->num_elements() * sizeof(Scalar)));
		}
	}

	Tensor<Scalar, Rank> &operator=(const Tensor<Scalar, Rank> &other) = delete;

	Tensor(const Tensor<Scalar, Rank> &other):
		refcount(other.refcount),
		data(other.get_data()),
		dev_data(other.get_dev_data()) {

		for (int i = 0; i < Rank; i++)
			this->dims[i] = other.dim(i);

		(*refcount)++;
	}

	~Tensor() {
		if (--(*this->refcount) == 0) {
			delete this->refcount;
			free(this->data);
			cudaErrchk(cudaFree(this->dev_data));
		}
	}

	/*
	 * This is a very shity version of
	 * reshape and does NOT do as much as
	 * numpy's reshape does. It works for us
	 * here though.
	 */
	Tensor<Scalar, Rank> reshape(std::initializer_list<int> dims) {
		assert(dims.size() == Rank);
		Tensor<Scalar, Rank> tensor(dims, false);
		tensor.refcount = this->refcount;
		tensor.data = this->data;
		tensor.dev_data = this->dev_data;
		(*tensor.refcount)++;
		assert(this->num_elements() == tensor.num_elements());
		return tensor;
	}

	void moveToDevice() {
		cudaErrchk(cudaMemcpy(this->dev_data, this->data,
			this->num_elements() * sizeof(Scalar), cudaMemcpyHostToDevice));
	}

	void moveToHost() {
		cudaErrchk(cudaMemcpy(this->data, this->dev_data,
			this->num_elements() * sizeof(Scalar), cudaMemcpyDeviceToHost));
	}

	void dump4D(FILE *f, const char* msg = "") {
		fprintf(f, "Tensor<%lld>[", Rank);
		for (int i = 0; i < Rank; i++)
			fprintf(f, i == Rank - 1 ? "%d" : "%d, ", this->dims[i]);
		fprintf(f, "]: %s\n", msg);

		for (int i = 0; i < this->dims[0]; i++) {
			fprintf(f, "Dim 0: %d:\n", i);
			for (int j = 0; j < this->dims[1]; j++) {
				fprintf(f, "Dim 1: %d:\n", j);
				for (int y = 0; y < this->dims[3]; y++) {
					for (int x = 0; x < this->dims[2]; x++) {
						fprintf(f, "\t%f", (*this)(i, j, x, y));
					}
					fprintf(f, "\n");
				}
				fprintf(f, "\n");
			}
			fprintf(f, "\n");
		}
	}
	
	void dump(FILE *f, const char* msg = "") {
		fprintf(f, "Tensor<%lld>[", Rank);
		for (int i = 0; i < Rank; i++)
			fprintf(f, i == Rank - 1 ? "%d" : "%d, ", this->dims[i]);
		fprintf(f, "]: %s\n", msg);

		for (int i = 0; i < this->num_elements(); i++)
			fprintf(f, i == this->num_elements() - 1 ? "%f\n" : "%f, ", this->data[i]);
	}

	void setConstant(Scalar value, bool set_gpu) {
		int n = this->num_elements();
		for (int i = 0; i < n; i++)
			this->data[i] = value;
		if (set_gpu)
			this->moveToDevice();
	}

	void setZero(bool set_gpu) {
		int n = this->num_elements();
		if (!set_gpu)
			this->setConstant(0, false);
		else
			cudaErrchk(cudaMemset(this->dev_data, 0, n * sizeof(Scalar)));
	}

	__host__ __device__
	int dim(int i) const {
		return this->dims[i];
	}

	__host__ __device__
	Scalar& operator() (int i, int j, int k, int l, int m) {
		static_assert(Rank == 5, "Wrong rank");
		int idx =
			i * this->dims[1] * this->dims[2] * this->dims[3] * this->dims[4] +
			j * this->dims[2] * this->dims[3] * this->dims[4] +
			k * this->dims[3] * this->dims[4] +
			l * this->dims[4] +
			m;
		#ifdef  __CUDA_ARCH__
		return this->dev_data[idx];
		#else
		return this->data[idx];
		#endif
	}

	__host__ __device__
	Scalar& operator() (int i, int j, int k, int l) {
		static_assert(Rank == 4, "Wrong rank");
		int idx =
			i * this->dims[1] * this->dims[2] * this->dims[3] +
			j * this->dims[2] * this->dims[3] +
			k * this->dims[3] +
			l;
		#ifdef  __CUDA_ARCH__
		return this->dev_data[idx];
		#else
		return this->data[idx];
		#endif
	}

	__host__ __device__
	Scalar& operator() (int i, int j) {
		static_assert(Rank == 2, "Wrong rank");
		int idx =
			i * this->dims[1] +
			j;
		#ifdef  __CUDA_ARCH__
		return this->dev_data[idx];
		#else
		return this->data[idx];
		#endif
	}

	__host__ __device__
	Scalar& operator() (int i) {
		static_assert(Rank == 1, "Wrong rank");
		#ifdef  __CUDA_ARCH__
		return this->dev_data[i];
		#else
		return this->data[i];
		#endif
	}

	__host__ __device__
	Scalar& flipped(int f, int c, int x, int y) {
		static_assert(Rank == 4, "Wrong rank");
		x = this->dims[2] - x - 1;
		y = this->dims[3] - y - 1;
		int idx =
			f * this->dims[1] * this->dims[2] * this->dims[3] +
			c * this->dims[2] * this->dims[3] +
			x * this->dims[3] +
			y;
		#ifdef  __CUDA_ARCH__
		return this->dev_data[idx];
		#else
		return this->data[idx];
		#endif
	}

	Scalar *get_data() const { return this->data; }
	Scalar *get_dev_data() const { return this->dev_data; }
	void set_data(Scalar *data) { this->data = data; }
	void set_dev_data(Scalar *dev_data) { this->dev_data = dev_data; }

	size_t num_elements() const {
		size_t res = 1;
		for (int i = 0; i < Rank; i++)
			res *= this->dims[i];

		return res;
	}

private:
	int dims[Rank];
	int *refcount;
	Scalar *data = nullptr;
	Scalar *dev_data = nullptr;
};

/*
 * Only the GPU data is copied, memory
 * allocated on the CPU is ignored!
 */
Tensor<float, 4> mergeAtChannelDim(Tensor<float, 4> &a, Tensor<float, 4> &b);

/*
 * Only the GPU data is copied, memory
 * allocated on the CPU is ignored!
 *
 * The channel at index @param channel is the first
 * channel dimension to end up in the second tensor.
 */
std::pair<Tensor<float, 4>, Tensor<float, 4>> splitAtChannelDim(Tensor<float, 4> &tensor, int channel);

