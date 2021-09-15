
# Repository for the HPC-Project 2021 by Kirill Menke and Lou Knauer

## U-Net CUDA Implementation:

The first thing that has to be done before using the U-Net is to generate the
required training data. For this, do the following:

```sh
cd ./data/cell-membranes
python3 ./png-to-text.py
```

Instead of reading the PNG files directly, we use a python script to transform the images
into text files and load them into our C++/Cuda application. On Linux, we used
images of size 512x512, whereas on Windows the default setting is only 256x256. This is because the GPUs on the
CIP machines provided by the FAU had more GPU memory compared to our private Windows device.
However, you can modify the default value in `./png-to-text.py` if you want to use bigger images on Windows as well 
(You will also need to edit `./gpu_unet/data.h`, line 13 in that case).


You can change the hyper-parameters of the network such as channels, layers, kernel/pool size, batch size, 
and the learning rate in the code of the the respective function you want to call (see below).

Every function will print a loss every few iterations and depending on the called function also a form of accuracy.
Most functions will at the end also print how long each layers forward/backward pass took.

### How to build and run on Linux
To build on Linux, go to the `./gpu_unet` folder. There, all the source code for the Cuda implementation
can be found. This directory also contains a `Makefile`. The default target builds the CNN,
so simply run `make -j 8` in that directory.

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

### How to build and run on Windows
On Windows, Visual Studio 2019 with its integrated compiler was used to compile and run the code.
You can open the provided solution file `./HPC.sln` within VS which includes the `gpu_unet` project 
and build it out of the box. The command line arguments which are passed via the CLI on Linux (see above),
can be specified in the project properties of `gpu_unet`: `Properties > Debugging > Command Arguments`.



## CNN C++ Implementation:

You can change all hyperparameters and the CNN architecture by modifying the
`./cpu_cnn/src/main.py` file.
The training data used by this CNN is MNIST and located in `data\mnist\mnist-train.txt`.
This is a textfile where each row corresponds to a flat 32 x 32 image.

### How to build and run on Linux
To build on Linux, go to the `./cpu_cnn` folder, which contains all the source code of the CPU CNN.
This folder also contains a `Makefile`. The default target builds the CNN,
so simply run `make -j 8` in that directory.

The binary will be `./cpu_cnn/main` and requires the number of iterations as its first argument:
```
# Runs the CNN with the MNIST dataset doing 1000 iterations
./main 1000
```


### How to build and run on Windows
On Windows, Visual Studio 2019 with its integrated compiler was used to compile and run the code.
You can open the provided solution file `./HPC.sln` within VS which includes the `cpu_cnn` project
and build it out of the box. The number of iterations which is passed via the CLI on Linux (see above),
can be specified in the project properties of `cpu_cnn`: `Properties > Debugging > Command Arguments`.