
#include <cuda.h>

#include "timer.h"

using namespace rfgpu;

Timer::Timer() {
    n_call = 0;
    t_total = 0.0;
    cudaEventCreate(&event0);
    cudaEventCreate(&event1);
}

Timer::~Timer() {
    cudaEventDestroy(event0);
    cudaEventDestroy(event1);
}

void Timer::start() {
    cudaEventRecord(event0);
}

void Timer::stop() {
    cudaEventRecord(event1);
    cudaEventSynchronize(event1);
    float t;
    cudaEventElapsedTime(&t, event0, event1);
    t_total += t;
    n_call++;
}

