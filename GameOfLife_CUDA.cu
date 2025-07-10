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
const unsigned int StartingArray_Columns = 14;
const unsigned int StartingArray_Rows = 18;

void runTest(int argc, char **argv);

// OpenCV Mat object would do this for me internaly, but to avoid
// a build dependency on OpenCV toolkit I just made my own simplified version here:
inline __device__ int IndxCalc(unsigned int Col_x, unsigned int Row_y){
    return (Row_y*StartingArray_Columns + Col_x);
}

extern "C"
void computeTick(unsigned int *reference, unsigned int *idata, const unsigned int len);


////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(unsigned int *g_idata, unsigned int *g_odata)
{
    // shared memory, just for demonstration (we don't really use it here)
    // the size is determined by the host application
    extern  __shared__  float sdata[]; // AyM: The size of this is determiend by the 3rd paramter of the Kernel invocation.

    // just checking, the number of threads in this block:
    // const unsigned int num_threads = blockDim.x*blockDim.y;

    // Calculate our thread id (x and y) for processing the array
    // Notice we are skipping the first and last row, and first and last columns,
    //  to avoid the boundary edges overruning.
    unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    // these _might_ print out in a certain order, but that ordering is _not guarenteed_ (its set by Warp Schedueler)
    printf("Tid %d[%d,%d]\n",IndxCalc(tid_x,tid_y),tid_x,tid_y); // Debug output: this is slow, "REMOVE BEFORE FLIGHT"

    sdata[IndxCalc(tid_x,tid_y)] = 0; // initialize to Zero

    // I _know_ this is bad for Warp Divergence, but there is no "else" path so it shouldn't be too costly...
    // This guards to make sure we are within the array boundaries, with a margin of 1 edge to avoid overrun.
    if ((tid_x > 0) && (tid_x < (StartingArray_Columns-1)) && (tid_y > 0) && (tid_y < (StartingArray_Rows -1)))
    {
        // Count the number of live neighbors, this is done into the temporary "Shared Memory" cache.

        // calculate "linearized" indicies into the array: Home, then North, South, East, West neighbors.
        // (using the OpenCV convention that 0,0 is upper-left of the array, and filled-out as right and then down...)
        // This is a "convolution kernel" operation and could be done as a loop, but I wanted to unroll it...

        // Note that we assume that g_idata is only 0 or 1 values for incrementing our counter correctly, this is enforced later
        // when we write from the SharedMemory sdata array into the g_odata with logical 0 or 1 only.
        sdata[IndxCalc(tid_x,tid_y)] += g_idata[IndxCalc(tid_x-1,tid_y-1)];
        sdata[IndxCalc(tid_x,tid_y)] += g_idata[IndxCalc(tid_x,tid_y-1)];
        sdata[IndxCalc(tid_x,tid_y)] += g_idata[IndxCalc(tid_x+1,tid_y-1)];

        sdata[IndxCalc(tid_x,tid_y)] += g_idata[IndxCalc(tid_x-1,tid_y)];
        sdata[IndxCalc(tid_x,tid_y)] += g_idata[IndxCalc(tid_x+1,tid_y)];
        
        sdata[IndxCalc(tid_x,tid_y)] += g_idata[IndxCalc(tid_x-1,tid_y+1)];
        sdata[IndxCalc(tid_x,tid_y)] += g_idata[IndxCalc(tid_x,tid_y+1)];
        sdata[IndxCalc(tid_x,tid_y)] += g_idata[IndxCalc(tid_x+1,tid_y+1)];
    } 
    //notice there is no "else" block, to avoid warp divergence

    __syncthreads(); // a barrier at which all threads in the Block must wait before any is allowed to proceed,
    // this ensures all of the shared memory (intermediate results) are "ready" before we continue to calculate the final output.

    // DEBUG TEST: just write intermediate data to global memory
    // g_odata[IndxCalc(tid_x,tid_y)] = (sdata[IndxCalc(tid_x,tid_y)]);
    g_odata[IndxCalc(tid_x,tid_y)] = (g_idata[IndxCalc(tid_x,tid_y)]);
    
    // TODO: test criteria based on "number of live neighbors" and write logical 0 or 1 for output:

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

    // This is the same starting array as my Python notebook, it will setup several repeating patterns.
    // total array size is 252, dimension are 18 rows x 14 columns.


    const unsigned int StartingArray_c[StartingArray_Columns*StartingArray_Rows] = 
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,1,1,1,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,1,1,0,0,0,0,0,0, \
    0,0,0,0,0,0,1,1,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,1,1,0,0,0,0, \
    0,0,0,0,0,0,0,0,1,1,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,1,1,1,0,0,0, \
    0,0,0,0,0,0,0,1,1,1,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
    0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    size_t mem_size_total = sizeof(unsigned int) * 18 * 14;

    // doing a check here becuase I was getting a compiler warning about memory sizes...
    printf("Memory Size total: %lu  (%u x %u x %lu)\n",mem_size_total,18,14,sizeof(unsigned int));

    // pointer to allocate host memory
    unsigned int *h_idata = 0;
    
    // allocate the host memory
    h_idata = (unsigned int *) malloc(mem_size_total);

    if (h_idata == 0){
        printf("host side malloc failed!\n");
        return;
    }

    // initalize the memory
    //memset(h_idata,0,mem_size_total);
    memcpy(h_idata,StartingArray_c,mem_size_total);
    
    // allocate device memory
    // Note I allocate this as a Linear array, but treat it as a 2d matrix later
    // TODO: use OpenCV Mat for this would be alot easier...
    unsigned int *d_idata;
    cudaMalloc((void **) &d_idata, mem_size_total);
    
    // copy host memory to device
    cudaMemcpy(d_idata, h_idata, mem_size_total,
                               cudaMemcpyHostToDevice);

    // allocate device memory for result
    unsigned int *d_odata;
    cudaMalloc((void **) &d_odata, mem_size_total);
    
    // check if memory created
    getLastCudaError("CUDA Malloc ");


    // initialize to all zeros, 
    // the output array must be initialized to Zero for the algorithm to work properly
    cudaMemset(d_odata,0,mem_size_total);

    // check if memory set
    getLastCudaError("CUDA Memset");

    // setup execution parameters
    dim3  grid(1, 1, 1); // simplifying assumption since our size is small: it all fits within one Grid block.

    // using x to represent which Column index within the Row,
    // and y to represent Row index. Internally the code in the Kernel
    // will guard against overruning the margins of the array.
    dim3  threads(StartingArray_Columns, StartingArray_Rows, 1);

    printf("Launching CUDA Kernel...\n");

    // execute the kernel. AyM Note: the 3rd parameter is Shared Memory allocation size for the CUDA block.
    // Given this only runs on Maxwell or higher architectures, I don't really need the Shared Memory for working space,
    //    but keeping it in here just as an example of how to use the feature. 
    // (My experience is that manually handling Shared Memory is unnecessary on Pascal or above,
    //      because the automatic cache does good enough.)
    testKernel<<< grid, threads, mem_size_total >>>(d_idata, d_odata);

    // make sure all kernels finished executing...
    cudaDeviceSynchronize();

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // allocate mem for the result on host side
    unsigned int *h_odata = 0;
    
    h_odata = (unsigned int *) malloc(mem_size_total);
    assert(h_odata);

    // copy result from device to host
    cudaMemcpy(h_odata, d_odata, mem_size_total,
                               cudaMemcpyDeviceToHost);

    // check if memcopy generated and error
    getLastCudaError("cudaMemcpyDeviceToHost");

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    for (int j = 0; j < StartingArray_Rows; j++) {
        // printf("Row %u: ",j);
        for (int i = 0; i < StartingArray_Columns; i++)
        {   
            unsigned int Indx = j*StartingArray_Columns + i;

            assert(Indx < mem_size_total);
            //printf(" %u (%u,%u) ",Indx,i,j);

            // basic ascii-art style printout of the resulting array.
            switch (h_odata[Indx]) {
                case 0:
                    printf(".");
                break;
                
                case 1:
                    printf("#");
                break;

                default:
                    printf("%u",h_odata[Indx]);
            }
        }   // end for j 
        printf("\n"); // CR/LF for the next row.
    } // end for i

    // compute reference solution
    unsigned int *reference = (unsigned int *) malloc(mem_size_total);
    computeTick(reference, h_idata, mem_size_total);

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
