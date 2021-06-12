
CXX=clang++
CXXFLAGS=-g -O2 -std=c++17 -Wall -Wno-unused-parameter -Wno-unused-private-field -Wno-delete-non-abstract-non-virtual-dtor -I./eigen-3.3.9/

HEADERS:=$(shell echo ./HPC/*.h)
CPPFILES:=$(filter-out ./HPC/NeuralNetwork.h, $(CPPFILES))
CPPFILES:=$(shell echo ./HPC/*.cpp)
CPPFILES:=$(filter-out ./HPC/NeuralNetwork.cpp, $(CPPFILES))
CPPFILES:=$(filter-out ./HPC/NeuralNetworkTests.cpp, $(CPPFILES))

.PHONY: all clean

all: main

./HPC/ReLU.o: ./HPC/ReLU.cpp ./HPC/ReLU.h
./HPC/MaxPool.o: ./HPC/MaxPool.cpp ./HPC/MaxPool.h
./HPC/SoftMax.o: ./HPC/SoftMax.cpp ./HPC/SoftMax.h
./HPC/Loss.o: ./HPC/Loss.cpp ./HPC/Loss.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

test: ./HPC/MaxPool.h ./HPC/ReLU.h ./HPC/SoftMax.h ./HPC/Loss.h ./HPC/Helper.h \
		./HPC/MaxPool.cpp ./HPC/ReLU.cpp ./HPC/SoftMax.cpp ./HPC/Loss.cpp ./HPC/Helper.cpp ./test.cpp
	$(CXX) $(CXXFLAGS) -o $@ ./HPC/MaxPool.cpp ./HPC/ReLU.cpp ./HPC/SoftMax.cpp ./HPC/Loss.cpp ./HPC/Helper.cpp ./test.cpp

main: $(HEADERS) $(CPPFILES)
	$(CXX) $(CXXFLAGS) -o $@ $(CPPFILES)

clean:
	rm -rf ./main
	rm -rf ./test
	rm -rf ./HPC/*.o


