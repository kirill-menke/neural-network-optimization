
# Repository for the HPC-Project 2021 by Kirill Menke and Lou Knauer

### How to use the CPU/C++ Implementation:

__*TODO!*__

### How to use the GPU/Cuda Implementation:

The first thing that has to be done after this repository was cloned is to generate the
required training data. For this, do the following:

```sh
cd ./data/cell-membranes
python3 ./png-to-text.py
```

Instead of reading the PNG files directly, we use a python script to transform the images
into text files which can be loaded into our C++/Cuda application. On Linux, you will use
images of size 512x512, on Windows by default only 256x256. The reason is that we tested
on the CIP machines privided by the University where the GPUs hat more than enough memory,
whereas the windows device used for development by Kirill has not that much. The python
script is very simple, look at it if you want to use bigger images on Windows as well
(You will also need to edit `./gpu_unet/data.h`, line 13 in that case).

After that, go to the `./gpu_unet` folder. There, all the source code for the Cuda implementation
can be found. This directory also contains a `Makefile`. The default target builds the CNN,
so simply run `make -j 8` in that directory. __TODO: Kirill, commite mal das CMake-Zeugs für Windows__.

The binary will be `./gpu_unet/test`. The first CLI argument it needs is the name of the CNN you want to run,
and the second one is how many training iterations you wish to do.

```
cd ./gpu_unet
make clean
make -j 8

# Runs code in the function main.cpp:bench_mini_unet.
# This is a very shallow Unet but at the end, timings are printed.
./test bench_unet 1000

# Runs code in the function main.cpp:test_unet.
# A larger Unet with more depth and layers and channels.
./test test_unet 100

# Runs code in the function main.cpp:bench_mnist.
# Images are only 32x32 in size, so not an ideal workload for the GPU.
# It shows that we implemented the layers generic enough for more than
# just one architecture.
./test mnist 10000

# Runs code in the function main.cpp:bench_conv.
# Only for testing, not very usable at anything.
./test bench_conv 1000

```

You can change hyper-parameters such as channels and layers and kernel/pool size
and the batch size and the learning rate right in the functions mentioned above.
Everything should be fairly self explainatory (We hope, sorry if it is not).

Every function will print the loss every few iterations and sometimes even some form of accuracy.
Most functions will at the end also print how long each layers forward/backward pass took.

