
CXX=clang++
CXXFLAGS=-std=c++17 -Wall -Wno-unused-parameter -Wno-unused-private-field -I./eigen-3.3.9/

HEADERS=$(shell find ./HPC -name "*.h")
CPPFILES=$(shell find ./HPC -name "*.cpp")

.PHONY: all clean

./HPC/ReLU.o: ./HPC/ReLU.cpp ./HPC/ReLU.h
./HPC/MaxPool.o: ./HPC/MaxPool.cpp ./HPC/MaxPool.h
./HPC/SoftMax.o: ./HPC/SoftMax.cpp ./HPC/SoftMax.h
./HPC/Loss.o: ./HPC/Loss.cpp ./HPC/Loss.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

all: main

test: ./HPC/MaxPool.h ./HPC/ReLU.h ./HPC/SoftMax.h ./HPC/Loss.h ./HPC/Helper.h \
		./HPC/MaxPool.cpp ./HPC/ReLU.cpp ./HPC/SoftMax.cpp ./HPC/Loss.cpp ./HPC/Helper.cpp ./test.cpp
	$(CXX) $(CXXFLAGS) -o $@ ./HPC/MaxPool.cpp ./HPC/ReLU.cpp ./HPC/SoftMax.cpp ./HPC/Loss.cpp ./HPC/Helper.cpp ./test.cpp


clean:
	rm -rf ./main
	rm -rf ./HPC/*.o


