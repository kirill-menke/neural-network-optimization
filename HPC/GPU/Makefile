#
# This Makefile is only for GPU-Specific Tests!
# Once Stuff works even a little bit, the files
# in this directory will be compiled by the root
# Makefile.
#

CUXX=nvcc
CUFLAGS=-g -std=c++14 --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored # -I../../eigen-3.3.9/

.PHONY: clean

test: ./test.cpp ./GPUConv.cu ./GPUConv.h ./Tensor.h
	$(CUXX) $(CUFLAGS) -o $@ ./test.cpp ./GPUConv.cu

clean:
	rm -rf ./test
