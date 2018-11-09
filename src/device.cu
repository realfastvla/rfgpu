#include <stdexcept>
#include <cuda.h>
#include "device.h"

using namespace rfgpu;

OnDevice::OnDevice(int device) {
    _device = device;
    _saved_device = -1;
    check_device();
}

void OnDevice::check_device() {
    int rv = cudaGetDevice(&_saved_device);
    if (rv != cudaSuccess)
        throw std::runtime_error("OnDevice::check_device error getting current device");
    if (_device==-1) {
        // No device specified yet, so use the current one
        _device = _saved_device;
    } else {
        // Change to the one we want
        int rv = cudaSetDevice(_device);
        if (rv != cudaSuccess)
            throw std::runtime_error("OnDevice::check_device error setting device");
    }
}

void OnDevice::reset_device() {
    int rv = cudaSetDevice(_saved_device);
    if (rv != cudaSuccess)
        throw std::runtime_error("OnDevice::reset_device error setting device");
}
