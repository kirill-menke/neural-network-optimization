#
# This Makefile is only for GPU-Specific Tests!
# Once Stuff works even a little bit, the files
# in this directory will be compiled by the root
# Makefile.
#

CUXX=nvcc
# CUFLAGS=-g -G -std=c++14 --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -I../../eigen-3.3.9/
CUFLAGS=-g -std=c++14 --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -I../../eigen-3.3.9/ -O3

.PHONY: clean

test: ./test.cpp ./GPUConv.cu ./GPUConv.h ./GPUMaxPool.cu ./GPUMaxPool.h ./GPUReLU.cu ./GPUReLU.h ./GPUSoftMax.cu ./GPUSoftMax.h ./GPUOptimizer.cu ./GPUUpsample.cu ./GPUUpsample.h ./Tensor.cu ./GPULoss.cu ./GPULoss.h ./GPUOptimizer.h ./Tensor.h ./utils.h ./utils.cpp
	$(CUXX) $(CUFLAGS) -o $@ ./test.cpp ./GPUConv.cu ./GPUMaxPool.cu ./GPUReLU.cu ./GPUOptimizer.cu ./GPUSoftMax.cu ./Tensor.cu ./GPUUpsample.cu ./GPULoss.cu ../SoftMax.cpp ../Loss.cpp ../Helper.cpp ./utils.cpp

clean:
	rm -rf ./test
