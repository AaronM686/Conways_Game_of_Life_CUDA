# Conways_Game_of_Life_CUDA
This is a C++ / CUDA implementation of "Conway's Game of Life" (Cellular Automaton). This is just a coding demonstration of how to do something in CUDA... don't read too much into this.

Yes I know that there are other CUDA implementations of this algorithm that are more advanced and claim to be very fast (some using lookup tables, etc). In this version, the eventual goal is to align towards the Nvidia Jetson architecture for Embedded devices, showing how the constraints of an embedded device drive the design choices. e.g. less Memory than a large RTX3090 would have, but you have Unified (Zero-Copy) memory, etc.

## Setup and Prerequisites:
You will need a Linux environment with CUDA and OpenCV. I'm using OpenCV becuase its a convenient "helper Library" to save images to disk and display an image to the screen. OpenCV Mat objects can be ingested into CUDA kernels, and the CUDA kernel results can be cast back to OpenCV::Mat objects for ease of use. The Jetson devices would likewise have a OpenCV4Tegra implementation you could leverage.

It is outiside the scope of this project to write a tutorial of how to get CUDA setup and runing on your system, there are plenty of other articles about that already. But I can sympathize, its definitely a hassle with multiple failure points along the way. Last I checked the Docker environments support for GPU acceleration were hit-or-miss. I Don't know about running this on Windows 10/11 but I heard the WSL might support GPU acceleration??? Given all that hassle, I can understand why alot of people just use a cloud solution from AWS or Azure instead, that's what Nvidia does for all of their online tutorial classes.

In the end, you need a system where you can run the "deviceQuery" example and get a successfull ouptut of CUDA enabled device.
here is the DeviceQuery output from my development machine (RTX 3090):
```
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 3090"
  CUDA Driver Version / Runtime Version          11.4 / 11.3
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 24235 MBytes (25411846144 bytes)
  (082) Multiprocessors, (128) CUDA Cores/MP:    10496 CUDA Cores
  GPU Max Clock rate:                            1740 MHz (1.74 GHz)
  Memory Clock rate:                             9751 Mhz
  Memory Bus Width:                              384-bit
  L2 Cache Size:                                 6291456 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.4, CUDA Runtime Version = 11.3, NumDevs = 1
Result = PASS
```
For convenience I am using my Desktop system with Ubuntu 18 (64-bit) to get an easier development/debugging environment, but I am writing the code in such a way it can easily be coppied-over to the Jetson system with no code changes:
```
$ lsb_release -a
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 18.04.6 LTS
Release:	18.04
Codename:	bionic

```
