#ifndef _TIMER_H
#define _TIMER_H

#include <string>

#include <cuda.h>

namespace rfgpu {

    class Timer
    {
        public:

            Timer();
            ~Timer();

            void start();
            void stop();

            double get_time() const { return t_total/(double)n_call; }

        protected:
            cudaEvent_t event0;
            cudaEvent_t event1;
            int n_call;
            double t_total;

    };
        
}

#endif // _TIMER_H
