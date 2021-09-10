
CXX=clang++
CXXFLAGS=-g -O2 -std=c++17 -Wno-unused-parameter -Wno-unused-private-field -Wno-delete-non-abstract-non-virtual-dtor -I./libs/eigen-3.3.9

HEADERS:=$(shell echo ./src/*.h)
CPPFILES:=$(shell echo ./src/*.cpp)
OBJFILES = $(patsubst %.cpp, %.o, $(patsubst %.cu, %.o, $(CPPFILES)))

.PHONY: all clean

all: main

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

main: $(HEADERS) $(OBJFILES)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJFILES)

clean:
	rm -rf ./main
	rm -rf ./src/*.o


