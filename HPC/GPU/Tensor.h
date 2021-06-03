#pragma once

#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <initializer_list>

#include "cuda-utils.h"

template<typename Scalar, size_t Rank>
class Tensor {
public:
	static constexpr int ON_CPU = 1;
	static constexpr int ON_GPU = 2;

	Tensor(int mode, std::initializer_list<int> dims) {
		assert(dims.size() == Rank);
		auto iter = dims.begin();
		this->total_size = 1;
		for (int i = 0; i < Rank; i++, iter++) {
			this->dims[i] = *iter;
			this->total_size *= *iter;
		}

		if ((mode & ON_CPU) != 0) allocCPU();
		if ((mode & ON_GPU) != 0) allocGPU();
	}

	void allocCPU() {
		if (this->data == nullptr)
			this->data = (Scalar *)calloc(this->total_size, sizeof(Scalar));
	}

	void allocGPU() {
		if (this->dev_data == nullptr)
			cudaErrchk(cudaMalloc(
				(void **)&this->dev_data,
				this->total_size * sizeof(Scalar)));
	}

	void moveToDevice() {
		allocGPU();
		cudaErrchk(cudaMemcpy(this->dev_data, this->data,
			this->total_size * sizeof(Scalar), cudaMemcpyHostToDevice));
	}

	void moveToHost() {
		allocCPU();
		cudaErrchk(cudaMemcpy(this->data, this->dev_data,
			this->total_size * sizeof(Scalar), cudaMemcpyDeviceToHost));
	}

	void free() {
		if (this->data != nullptr) ::free(this->data);
		if (this->dev_data != nullptr) cudaErrchk(cudaFree(this->dev_data));
		this->data = nullptr;
		this->dev_data = nullptr;
	}

	void dump(FILE *f, const char* msg = "") {
		fprintf(f, "Tensor<%lld>[", Rank);
		for (int i = 0; i < Rank; i++)
			fprintf(f, i == Rank - 1 ? "%d" : "%d, ", this->dims[i]);
		fprintf(f, "]: %s\n", msg);

		if (Rank != 4) {
			for (int i = 0; i < this->total_size; i++)
				fprintf(f, i == this->total_size - 1 ? "%f\n" : "%f, ", this->data[i]);
			return;
		}

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

	__host__ __device__
	int dim(int i) const {
		return this->dims[i];
	}

	__host__ __device__
	Scalar& operator() (int i, int j, int k, int l) {
		static_assert(Rank == 4);
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
	Scalar& flipped(int f, int c, int x, int y) {
		static_assert(Rank == 4);
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

	Scalar *get_data() { return this->data; }
	Scalar *get_dev_data() { return this->dev_data; }
	void set_data(Scalar *data) { this->data = data; }
	void set_dev_data(Scalar *dev_data) { this->dev_data = dev_data; }

	size_t num_elements() {
		return this->total_size;
	}

private:
	int dims[Rank];
	size_t total_size = 0;
	Scalar *data = nullptr;
	Scalar *dev_data = nullptr;
};

