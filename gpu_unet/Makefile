CUXX    = nvcc
CUFLAGS = --compiler-options -Wall --compiler-options -Wextra -std=c++14 --expt-relaxed-constexpr

.PHONY: clean all

ifdef DEBUG_BUILD
CUFLAGS += -g -G
else
CUFLAGS += -g -O3 --use_fast_math
endif

SRCFILES = $(shell find . -name '*.cpp' -or -name '*.cu')
OBJFILES = $(patsubst %.cpp, %.o, $(patsubst %.cu, %.o, $(SRCFILES)))

all: test

%.o: %.cpp
	$(CUXX) $(CUFLAGS) -o $@ -c $<

%.o: %.cu
	$(CUXX) $(CUFLAGS) -o $@ -c $<

./activations.o: ./activations.cu ./activations.h ./tensor.h
./conv.o:        ./conv.cu        ./conv.h        ./tensor.h ./optimizer.h
./data.o:        ./data.cpp       ./data.h        ./tensor.h
./loss.o:        ./loss.cu        ./loss.h        ./tensor.h
./optimizer.o:   ./optimizer.cu   ./optimizer.h   ./tensor.h
./pooling.o:     ./pooling.cu     ./pooling.h     ./tensor.h
./utils.o:       ./utils.cpp      ./utils.h       ./tensor.h
./tensor.o:      ./tensor.cu      ./tensor.h

./conv-relu.o:    ./conv-relu.cu    ./conv-relu.h    ./tensor.h ./optimizer.h
./conv-softmax.o: ./conv-softmax.cu ./conv-softmax.h ./tensor.h ./optimizer.h
./unet-down.o:    ./unet-down.cu    ./unet-down.h    ./tensor.h
./unet-up.o:      ./unet-up.cu      ./unet-up.h      ./tensor.h

./test.o: $(shell find . -name '*.h')

test: $(OBJFILES)
	@mkdir -p ./weights/unet ./weights/mnist
	$(CUXX) $(CUFLAGS) -o $@ $^

clean:
	rm -rf ./test
	rm -rf *.o
	rm -rf *.o
