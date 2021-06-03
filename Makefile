
CXX=clang++
CXXFLAGS=-std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-private-field -I./eigen-3.3.9/

HEADERS=$(shell find ./HPC -name "*.h")
CPPFILES=$(shell find ./HPC -name "*.cpp")

.PHONY: all clean

all: main

# Not very good, but it works:
main: $(HEADERS) $(CPPFILES)
	$(CXX) $(CXXFLAGS) -o $@ $(CPPFILES)

clean:
	rm -rf ./main
	rm -rf ./HPC/*.o


