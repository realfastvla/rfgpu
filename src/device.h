#ifndef _DEVICE_H
#define _DEVICE_H

#include <cuda.h>

namespace rfgpu {

    /* This generic base class associates an instance with a specific
     * CUDA device.  Derived classes are responsible for calling
     * check_device() and reset_device() as needed to make sure the correct
     * device is currently selected and gets reverted when done.
     */
    class OnDevice
    {
        protected:
            // The constructor should set the device to the requested one
            // (or current in default arg case) so that any arrays or other
            // resources allocated in the derived class constructors get
            // attached to the correct device.
            OnDevice(int device=-1);

            // Save the current device, set to the stored device.
            void check_device();

            // Reset back to saved device
            void reset_device();

            int _saved_device;
            int _device;

            friend class CheckDevice;
    };

    /* Create a CheckDevice instance within any method that needs to
     * switch to the device, then reset when exiting.
     */
    class CheckDevice
    {
        public:
            CheckDevice(OnDevice *_od) { od=_od; _od->check_device(); }
            ~CheckDevice() { od->reset_device(); }
        protected:
            OnDevice *od;
    };
}

#endif
