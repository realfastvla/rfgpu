#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>

#include "image.h"

using namespace rfgpu;

Image::Image() {
    plan = NULL; 
    xpix = ypix = 0;
}

Image::~Image() {
    if (plan) cufftDestroy(plan);
}

void Image::setup() {
    cufftPlan2d(&plan, xpix, ypix, CUFFT_C2R); // TODO check for error
}

void Image::operate(cufftComplex *vis, cufftReal *img) {
    cufftExecC2R(plan, vis, img);
}

