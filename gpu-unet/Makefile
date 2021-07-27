
CUXX=nvcc
# CUFLAGS=-g -G --compiler-options -Wall --compiler-options -Wextra -std=c++14
CUFLAGS=-g -O2 --use_fast_math --compiler-options -Wall --compiler-options -Wextra -std=c++14 --expt-relaxed-constexpr

.PHONY: clean all

SRCFILES = $(shell find . -name '*.cpp' -or -name '*.cu')
OBJFILES = $(patsubst %.cpp, %.o, $(patsubst %.cu, %.o, $(SRCFILES)))

all: test

%.o: %.cpp
	$(CUXX) $(CUFLAGS) -o $@ -c $<

%.o: %.cu
	$(CUXX) $(CUFLAGS) -o $@ -c $<

test: $(OBJFILES)
	@mkdir -p ./weights/unet
	$(CUXX) $(CUFLAGS) -o $@ $^

clean:
	rm -rf ./test
	rm -rf *.o
	rm -rf *.o
