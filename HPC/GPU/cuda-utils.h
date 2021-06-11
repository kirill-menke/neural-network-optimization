#pragma once

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define cudaErrchk(code) cudaErrCheck((code), __FILE__, __LINE__)
static inline void cudaErrCheck(cudaError_t code, const char file[], int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "Cuda Error: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

static inline dim3 getBlockDim(int w, int h, int n) {
	dim3 blockDim(16, 16, 1);
	return blockDim;
}

static inline dim3 getGridDim(int w, int h, int n) {
	dim3 gridDim((w + 16 - 1) / 16, (h + 16 - 1) / 16, n);
	return gridDim;
}


