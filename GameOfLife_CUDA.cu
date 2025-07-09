////////////////////////////////////////////////////////////////////////////
//
// CUDA implementation of "Conway's Game of Life" cellular automaton.
//   https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
//
// This is a coding skills demonstration created by Aaron Mosher.
// https://github.com/AaronM686
//
// Makefile and boilerplate support code is based on Nvidia samples "Template"
// You will need the Samples directory to compile this, since
// I rely on several helper-functions they provide to streamline the code.
// 
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>  // from the "/common/inc/" folder of the Nvidia CUDA samples.
#include <helper_functions.h> // helper functions for SDK examples

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"
void computeTick(float *reference, float *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(float *g_idata, float *g_odata)
{
    // shared memory, just for demonstration (we don't really use it here)
    // the size is determined by the host application
    extern  __shared__  float sdata[]; // AyM: The size of this is determiend by the 3rd paramter of the Kernel invocation.

    // access number of threads in this block
    const unsigned int num_threads = blockDim.x; // AyM: Need to update this for a 2-dimensional thread block.

    // Calculate our thread id
    const unsigned int tid = num_threads*threadIdx.y + threadIdx.x; // AyM: Need to update this for a 2-dimensional thread block.

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads(); // a barrier at which all threads in the Block must wait before any is allowed to proceed

    // these _might_ print out in a certain order, but that ordering is _not guarenteed_ (its set by Warp Schedueler)
    printf("Tid %d (%d,%d,%d)\n",tid,threadIdx.x,threadIdx.y,threadIdx.z); // Debug output: this is slow, need to comment-out later.

    // TODO: perform your computations here !
    
    

    __syncthreads(); // a barrier at which all threads in the block must wait before any is allowed to proceed
    
    // note: you must have the synctreads() barrier if you are going to use SharedMemory block, to ensure all processing is done.

    // write data to global memory
    g_odata[tid] = sdata[tid];
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
// AyM: the main Run function for the iteration of "Conway's game of life"
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResult = true;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);
    // AyM NOTE: This requires the "helper_cuda.h" from the "/common/" folder of the Nvidia CUDA samples.

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    unsigned int array_dimension = 4; // AyM: using this a dimensions for a 2-dimensional thread block.
    size_t mem_size_total = sizeof(float) * array_dimension * array_dimension;

    // doing a check here becuase I was getting a compiler warning about memory sizes...
    printf("Memory Size total: %lu  (%u x %u x %lu)\n",mem_size_total,array_dimension,array_dimension,sizeof(float));

    // pointer to allocate host memory
    float *h_idata = 0;
    
    // allocate the host memory
    h_idata = (float *) malloc(mem_size_total);

    if (h_idata == 0){
        printf("host side malloc failed!\n");
        return;
    }

    // initalize the memory
    memset(h_idata,0,mem_size_total);
    
    // allocate device memory
    // Note I allocate this as a Linear array, but treat it as a 2d matrix later
    // TODO: use OpenCV Mat for this would be alot easier...
    float *d_idata;
    cudaMalloc((void **) &d_idata, mem_size_total);
    
    // copy host memory to device
    cudaMemcpy(d_idata, h_idata, mem_size_total,
                               cudaMemcpyHostToDevice);

    // allocate device memory for result
    float *d_odata;
    cudaMalloc((void **) &d_odata, mem_size_total);
    
    // check if memory created
    getLastCudaError("CUDA Malloc ");


    // initialize to all zeros, 
    // this must be initialized to Zero for the algorithm to work properly
    cudaMemset(d_odata,0,mem_size_total);

    // setup execution parameters
    dim3  grid(1, 1, 1);
    dim3  threads(array_dimension, array_dimension, 1);

    // check if memory set
    getLastCudaError("CUDA Memset");


    printf("Launching CUDA Kernel...\n");

    // execute the kernel. AyM Note: the 3rd parameter is Shared Memory allocation size for the CUDA block.
    // Given this only runs on Maxwell or higher architectures, do I really need the Shared Memory anymore?
    testKernel<<< grid, threads, mem_size_total >>>(d_idata, d_odata);

    // make sure all kernels finished executing...
    cudaDeviceSynchronize();

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // allocate mem for the result on host side
    float *h_odata = (float *) malloc(mem_size_total);
    // copy result from device to host
    cudaMemcpy(h_odata, d_odata, sizeof(float) * mem_size_total,
                               cudaMemcpyDeviceToHost);

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // compute reference solution
    float *reference = (float *) malloc(mem_size_total);
    computeTick(reference, h_idata, array_dimension);

    // check result
    //if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
    //{
    //    // write file for regression test
    //    sdkWriteFile("./data/regression.dat", h_odata, num_threads, 0.0f, false);
    //}
    //else
    //{
    //    // custom output handling when no regression test running
    //    // in this case check if the result is equivalent to the expected solution
    //    bTestResult = compareData(reference, h_odata, num_threads, 0.0f, 0.0f);
    //}

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(reference);
    cudaFree(d_idata);
    cudaFree(d_odata);

    printf("Conways_Game_of_Life_CUDA Done.\n");
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
