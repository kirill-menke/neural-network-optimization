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
	dim3 blockDim(64, 64, 1);
	return blockDim;
}

static inline dim3 getGridDim(int w, int h, int n) {
	dim3 gridDim((w + 64 - 1) / 64, (h + 64 - 1) / 64, n);
	return gridDim;
}


