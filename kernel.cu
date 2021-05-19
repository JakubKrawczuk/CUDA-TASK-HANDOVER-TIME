
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>

#define TEST_RUN_COUNT 10
#define IGNORE_FIRST_TESTS 2
#define VERBOSE false

cudaError_t addWithCuda(float *c, float*a, float *b, unsigned int size);

__global__ void addKernel(float *c, float *a, float *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void addKernelBlock(float* c, float* a, float* b) {
    int ti = threadIdx.x;
    int bi = blockIdx.x;
    int bd = blockDim.x;
    int i = bd * bi + ti;
    c[i] = a[i] + b[i];
}

void generateData(float* arr, int arraySize, float mul);

float minMaxAvg(char op, long long time) {
    static int no = 0;
    static long long min;
    static long long max;
    static long long sum=0;

    if (op == 'r') {
        //reset
        no = 0;
        sum = 0;
    }
    else if (op == 'e') {
        printf("\tmin: %d\n", min);
        printf("\tmax: %d\n", max);
        float avg = (float)sum / (TEST_RUN_COUNT - IGNORE_FIRST_TESTS);
        printf("\tavg: %f\n", avg);
        return avg;
    }
    else {
        if (no == 0) {
            min = time;
            max = time;
        }
        else {
            if (min > time) min = time;
            if (max < time) max = time;
        }

        if (no >= IGNORE_FIRST_TESTS) {
            sum += time;
        }        

        if(VERBOSE) printf("\t\t%d\n", time);
        no++;
    }
    return 0;
}

int main()
{
    const int arraySize = 50000; //test value 100 and 50000
    float a[arraySize] = { 0 };
    float b[arraySize] = { 0 };
    float c[arraySize] = { 0 };

    generateData(a, arraySize, 1);
    generateData(b, arraySize, 0.5);

    /*
    * Time overhead caused by getting time and first variable declaration
    * In my case this value veries from 0 to 300 ns
    */
    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();
    /*printf("Timer overhead check 1: %d ns (likely 0-300ns)\n",
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());

    printf("Timer overhead check 2 ns (likely 0-300ns)\n");
    */

    //time overhead caused by getting time (also varies from 0 to 300 ns)
    if (VERBOSE) {
        for (int i = 0; i < TEST_RUN_COUNT; i++) {
            start = std::chrono::high_resolution_clock::now();
            finish = std::chrono::high_resolution_clock::now();
            minMaxAvg('-', std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
        }
        minMaxAvg('e', 0);
        minMaxAvg('r', 0);
    }

    /*
    //printf overhead
    start = std::chrono::high_resolution_clock::now();
    printf("Fake value only for test: %d ns\n",
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
    finish = std::chrono::high_resolution_clock::now();
    printf("Printf overhead: %d ns (likely ~32000ns)\n",
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
    */

    /*
    * CUDA
    */
    printf("\n");
    // Add vectors in parallel.

    //start = std::chrono::high_resolution_clock::now();
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    //finish = std::chrono::high_resolution_clock::now();

    /*printf("CUDA in %d ns - contains additional overhead from printf and time check look at partial values\n",
       std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
    */

    /*
    printf("\n");
    for (int i = 0; i < arraySize; i++) {
        printf("%f; ", c[i]);
    }
    printf("\n");
    */

    /*
    * CPU
    */
    printf("CPU in ns\n");
    for (int i = 0; i < TEST_RUN_COUNT; i++) {
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < arraySize; i++) {
            c[i] = a[i] + b[i];
        }
        finish = std::chrono::high_resolution_clock::now();
        minMaxAvg('-', std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
    }
    minMaxAvg('e', TEST_RUN_COUNT);
    minMaxAvg('r', 0);

    printf("\n\nFOR 100 elements\n");
    //FOR 100
    cudaStatus = addWithCuda(c, a, b, 100);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("CPU in ns\n");
    for (int i = 0; i < TEST_RUN_COUNT; i++) {
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            c[i] = a[i] + b[i];
        }
        finish = std::chrono::high_resolution_clock::now();
        minMaxAvg('-', std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
    }
    minMaxAvg('e', TEST_RUN_COUNT);
    minMaxAvg('r', 0);
    

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float*c, float *a, float *b, unsigned int size)
{
    cudaError_t cudaStatus;
    float timeSum = 0;

    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();

    // Choose which GPU to run on, change this on a multi-GPU system.
    printf("CUDA - device choosed ns\n");
    for (int i = 0; i < TEST_RUN_COUNT; i++) {
        start = std::chrono::high_resolution_clock::now();
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }
        finish = std::chrono::high_resolution_clock::now();
        minMaxAvg('-', std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
    }
    timeSum += minMaxAvg('e', 0);
    minMaxAvg('r', 0);

    

    /*
    * MEMORY
    */
    printf("CUDA - memory prepare in ns) \n");
    for (int i = 0; i < TEST_RUN_COUNT; i++) {
        start = std::chrono::high_resolution_clock::now();
        float* dev_a = 0;
        float* dev_b = 0;
        float* dev_c = 0;

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        finish = std::chrono::high_resolution_clock::now();
        minMaxAvg('-', std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());

        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
    }
    timeSum += minMaxAvg('e', 0);
    minMaxAvg('r', 0);
    
    //for real this time
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    

    /*
    * Claculations
    */
    printf("CUDA - calculations in ns\n");
    for (int i = 0; i < TEST_RUN_COUNT; i++) {
        start = std::chrono::high_resolution_clock::now();

        if (size <= 256) {
            //one Block
            // Launch a kernel on the GPU with one thread for each element.
            addKernel << <1, size >> > (dev_c, dev_a, dev_b);
        }
        else {
            //multiple blocks
            int blocks_count = size / 256 + ((size % 256 == 0) ? 0 : 1);
            addKernelBlock << <blocks_count, 256 >> > (dev_c, dev_a, dev_b);
        }

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        finish = std::chrono::high_resolution_clock::now();
        minMaxAvg('-', std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
    }
    timeSum += minMaxAvg('e', 0);
    minMaxAvg('r', 0);
       


    /*
    * Value return
    */
    printf("CUDA - return in ns\n");
    for (int i = 0; i < TEST_RUN_COUNT; i++) {
        start = std::chrono::high_resolution_clock::now();

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        finish = std::chrono::high_resolution_clock::now();
        minMaxAvg('-', std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
    }
    timeSum += minMaxAvg('e', 0);
    minMaxAvg('r', 0);

    printf("CUDA - all avg: %f ns\n", timeSum);
       

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

void generateData(float* arr, int arraySize, float mul) {
    for (int i = 0; i < arraySize; i++) {
        arr[i] = (i + 1) * mul;
    }
}