#
# This Makefile is only for GPU-Specific Tests!
# Once Stuff works even a little bit, the files
# in this directory will be compiled by the root
# Makefile.
#

CUXX=nvcc
# CUFLAGS=-g -G -O2 -std=c++14 --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -I../../eigen-3.3.9/
CUFLAGS=-std=c++14 --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -I../../eigen-3.3.9/ -O3

.PHONY: clean

test: ./test.cpp ./GPUConv.cu ./GPUConv.h ./GPUMaxPool.cu ./GPUMaxPool.h ./GPUReLU.cu ./GPUReLU.h ./GPUSoftMax.cu ./GPUSoftMax.h ./GPUOptimizer.cu ./GPUOptimizer.h ./Tensor.h
	$(CUXX) $(CUFLAGS) -o $@ ./test.cpp ./GPUConv.cu ./GPUMaxPool.cu ./GPUReLU.cu ./GPUOptimizer.cu ./GPUSoftMax.cu ../SoftMax.cpp ../Loss.cpp ../Helper.cpp

clean:
	rm -rf ./test
