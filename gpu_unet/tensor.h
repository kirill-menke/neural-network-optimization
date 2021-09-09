#pragma once

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <initializer_list>
#include <utility>

/*
 * Wenn 1, dann wird via `assert()`
 * geprueft ob Zugriff gültig.
 */
#define TENSOR_DIMCHECK 0

#define cudaErrchk(code) cudaErrCheck((code), __FILE__, __LINE__)
static inline void cudaErrCheck(cudaError_t code, const char file[], int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "Cuda Error: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
		exit(1);
	}
}

/*
 * Hilfs-Funktion zum ausrechnen von Grid/Blocks.
 */
static inline void getGridSize(dim3 &gridDim, dim3 &blockDim, int channels, int width, int height, int xytpb = 16) {
	blockDim.x = xytpb;
	blockDim.y = xytpb;
	blockDim.z = 1;
	gridDim.x = (height + blockDim.x - 1) / blockDim.x;
	gridDim.y = (width  + blockDim.y - 1) / blockDim.y;
	gridDim.z = channels;
	//  printf("blockDim(%d, %d, %d), gridDim(%d, %d, %d)\n",
	//	blockDim.x, blockDim.y, blockDim.z,
	//	gridDim.x, gridDim.y, gridDim.z);
}

/*
 * Siehe tensor.cu, cudaMalloc ist teuer, daher
 * werden von einer hashmap in tensor.cu aus
 * Speicherbereiche direkt wiederverwendet.
 */
void *alloc_gpu_memory(size_t bytes);
void free_gpu_memory(void *ptr, size_t bytes);

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
		for (size_t i = 0; i < Rank; i++, iter++)
			this->dims[i] = *iter;

		if (do_allocs) {
			this->refcount = new int(1);
			this->dev_data = (Scalar *)alloc_gpu_memory(this->num_elements() * sizeof(Scalar));
			this->data = (Scalar *)malloc(this->num_elements() * sizeof(Scalar));
		}
	}

	Tensor(int batch_size, int channels, int width, int height):
		Tensor({ batch_size, channels, width, height }) {}

	Tensor(int n, int m):
		Tensor({ n, m }) {}

	Tensor(int n):
		Tensor({ n }) {}

#if 1
	Tensor<Scalar, Rank>& operator=(const Tensor<Scalar, Rank>& other) = delete;
#else
	Tensor<Scalar, Rank>& operator=(const Tensor<Scalar, Rank>& other) {
		assert(get_data() != other.get_data() && get_dev_data() != other.get_dev_data());
		if (--(*refcount) == 0) {
			delete this->refcount;
			free(this->data);
			free_gpu_memory(this->dev_data, this->num_elements() * sizeof(Scalar));
		}

		this->refcount = other.refcount;
		this->data = other.get_data();
		this->dev_data = other.get_dev_data();
		for (size_t i = 0; i < Rank; i++)
			this->dims[i] = other.dim(i);

		(*refcount)++;
		return *this;
	}
#endif

	template<size_t NewRank>
	Tensor<Scalar, NewRank> reshape(std::initializer_list<int> new_dims) const {
		size_t n = 1;
		for (auto iter = new_dims.begin(); iter != new_dims.end(); iter++)
			n *= *iter;

		assert(new_dims.size() == NewRank && n == this->num_elements());

		Tensor<Scalar, NewRank> reshaped(new_dims, false);
		reshaped.refcount = refcount;
		reshaped.data = get_data();
		reshaped.dev_data = get_dev_data();
		(*refcount)++;
		return reshaped;
	}

	/*
	 * Das ist nur eine shallow copy, d.h.
	 * es werden KEINE Daten kopiert. Der neue
	 * Tensor zeigt intern auf genau die selben Daten wie vorher,
	 * verändert man die einen Daten ändern sie die beim via Copy-
	 * Constructor kopierten auch!
	 */
	Tensor(const Tensor<Scalar, Rank> &other) :
		dev_data(other.get_dev_data()),
		data(other.get_data()),
		refcount(other.refcount) {

		for (size_t i = 0; i < Rank; i++)
			this->dims[i] = other.dim(i);

		(*refcount)++;
	}

	~Tensor() {
		if (--(*this->refcount) == 0) {
			delete this->refcount;
			free(this->data);
			free_gpu_memory(this->dev_data, this->num_elements() * sizeof(Scalar));
		}
	}

	void moveToDevice() const {
		cudaErrchk(cudaMemcpy(this->dev_data, this->data,
			this->num_elements() * sizeof(Scalar), cudaMemcpyHostToDevice));
	}

	const Tensor<Scalar, Rank>& moveToHost() const {
		cudaErrchk(cudaMemcpy(this->data, this->dev_data,
			this->num_elements() * sizeof(Scalar), cudaMemcpyDeviceToHost));

		return *this;
	}

	/*
	 * Nullt nur den GPU-Speicher!
	 */
	void setZero() {
		int n = this->num_elements();
		cudaErrchk(cudaMemset(this->dev_data, 0, n * sizeof(Scalar)));
	}


	void setConstant(Scalar value) {
		int n = this->num_elements();
		cudaErrchk(cudaMemset(this->dev_data, value, n * sizeof(Scalar)));
	}


	void dump4D(FILE *f, const char* msg = "") const {
		static_assert(Rank == 4, "Wrong rank");
		fprintf(f, "Tensor<%lu>[", Rank);
		for (size_t i = 0; i < Rank; i++)
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

	void dump2D(FILE *f, const char* msg = "") const {
		static_assert(Rank == 2, "Wrong rank");
		fprintf(f, "Tensor<%lu>[", Rank);
		for (size_t i = 0; i < Rank; i++)
			fprintf(f, i == Rank - 1 ? "%d" : "%d, ", this->dims[i]);
		fprintf(f, "]: %s\n", msg);

		for (int y = 0; y < this->dims[1]; y++) {
			for (int x = 0; x < this->dims[0]; x++) {
				fprintf(f, "\t%f", (*this)(x, y));
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}

	void dump(FILE *f, const char* msg = "") const {
		fprintf(f, "Tensor<%lu>[", Rank);
		for (size_t i = 0; i < Rank; i++)
			fprintf(f, i == Rank - 1 ? "%d" : "%d, ", this->dims[i]);
		fprintf(f, "]: %s\n", msg);

		for (size_t i = 0; i < this->num_elements(); i++)
			fprintf(f, i == this->num_elements() - 1 ? "%f\n" : "%f, ", this->data[i]);

		fprintf(f, "\n");
	}

	__host__ __device__
	int dim(int i) const {
#if TENSOR_DIMCHECK
		assert(0 <= i && i < int(Rank));
#endif
		return this->dims[i];
	}

	__host__ __device__
	Scalar& operator() (int i, int j, int k, int l, int m, int n) {
		static_assert(Rank == 6, "Wrong rank");
#if TENSOR_DIMCHECK
		assert(0 <= i && i < dims[0]);
		assert(0 <= j && j < dims[1]);
		assert(0 <= k && k < dims[2]);
		assert(0 <= l && l < dims[3]);
		assert(0 <= m && m < dims[4]);
		assert(0 <= n && n < dims[5]);
#endif
		int idx = n + dims[5] * (m + dims[4] * (l + dims[3] * (k + dims[2] * (j + dims[1] * (i)))));
		#ifdef  __CUDA_ARCH__
		return this->dev_data[idx];
		#else
		return this->data[idx];
		#endif
	}

	__host__ __device__
	Scalar& operator() (int i, int j, int k, int l, int m) {
		static_assert(Rank == 5, "Wrong rank");
#if TENSOR_DIMCHECK
		assert(0 <= i && i < dims[0]);
		assert(0 <= j && j < dims[1]);
		assert(0 <= k && k < dims[2]);
		assert(0 <= l && l < dims[3]);
		assert(0 <= m && m < dims[4]);
#endif
		int idx = m + dims[4] * (l + dims[3] * (k + dims[2] * (j + dims[1] * (i))));
		#ifdef  __CUDA_ARCH__
		return this->dev_data[idx];
		#else
		return this->data[idx];
		#endif
	}

	__host__ __device__
	inline Scalar& operator() (int i, int j, int k, int l) {
		static_assert(Rank == 4, "Wrong rank");
#if TENSOR_DIMCHECK
		assert(0 <= i && i < dims[0]);
		assert(0 <= j && j < dims[1]);
		assert(0 <= k && k < dims[2]);
		assert(0 <= l && l < dims[3]);
#endif
		int idx = l + dims[3] * (k + dims[2] * (j + dims[1] * (i)));
		#ifdef  __CUDA_ARCH__ 
		return this->dev_data[idx];
		#else
		return this->data[idx];
		#endif
	}

	__host__ __device__
	Scalar operator() (int i, int j, int k, int l) const {
		static_assert(Rank == 4, "Wrong rank");
#if TENSOR_DIMCHECK
		assert(0 <= i && i < dims[0]);
		assert(0 <= j && j < dims[1]);
		assert(0 <= k && k < dims[2]);
		assert(0 <= l && l < dims[3]);
#endif
		int idx = l + dims[3] * (k + dims[2] * (j + dims[1] * (i)));
		#ifdef  __CUDA_ARCH__
		return this->dev_data[idx];
		#else
		return this->data[idx];
		#endif
	}

	__host__ __device__
		Scalar& operator() (int i, int j, int k) {
		static_assert(Rank == 3, "Wrong rank");
		int idx =
			i * this->dims[1] * this->dims[2] +
			j * this->dims[2] +
			k;
		#ifdef  __CUDA_ARCH__
		return this->dev_data[idx];
		#else
		return this->data[idx];
		#endif
	}

	__host__ __device__
	Scalar& operator() (int i, int j) {
		static_assert(Rank == 2, "Wrong rank");
		assert(0 <= i && i < dims[0]);
		assert(0 <= j && j < dims[1]);
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
	Scalar operator() (int i, int j) const {
		static_assert(Rank == 2, "Wrong rank");
		assert(0 <= i && i < dims[0]);
		assert(0 <= j && j < dims[1]);
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
#if TENSOR_DIMCHECK
		assert(0 <= i && i < dims[0]);
#endif
		#ifdef  __CUDA_ARCH__
		return this->dev_data[i];
		#else
		return this->data[i];
		#endif
	}

	Scalar *get_data() const { return this->data; }
	Scalar *get_dev_data() const { return this->dev_data; }
	void set_data(Scalar *data) { this->data = data; }
	void set_dev_data(Scalar *dev_data) { this->dev_data = dev_data; }

	size_t num_elements() const {
		size_t res = 1;
		for (size_t i = 0; i < Rank; i++)
			res *= this->dims[i];

		return res;
	}

private:
	/*
	 * Achtung: Die eigentlichen Regeln für das verwenden
	 * des `restrict` qualifiers erfüllen wir NICHT, weil
	 * es ja mehrere Tensoren geben kann die die selben Pointer auf
	 * die Daten beinhalten. Man sollte aber nie einem Cuda-Kernel oder
	 * einer Methode 2x Tensoren übergeben müssen die auf das selbe zeigen,
	 * deshalb sollte das bei uns keine Probleme machen.
	 */
	Scalar * __restrict__ dev_data = nullptr;
	Scalar * __restrict__ data = nullptr;
	int *refcount;
	int dims[Rank];
};

/*
 * Only the GPU data is copied, memory
 * allocated on the CPU is ignored!
 */
Tensor<float, 4> concat(const Tensor<float, 4> &a, const Tensor<float, 4> &b);

/*
 * Only the GPU data is copied, memory
 * allocated on the CPU is ignored!
 *
 * The channel at index @param channel is the first
 * channel dimension to end up in the second tensor.
 */
std::pair<Tensor<float, 4>, Tensor<float, 4>> split(const Tensor<float, 4> &tensor, int channel);

/*
 * Add two Tensors on the GPU
 */
Tensor<float, 4> operator+(const Tensor<float, 4> &a, const Tensor<float, 4> &b);



