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

            double get_time_percall() const { return 1e3*t_total/(double)n_call; }
            double get_time_total() const { return 1e3*t_total; }

        protected:
            cudaEvent_t event0;
            cudaEvent_t event1;
            int n_call;
            double t_total;

    };
        
}

#endif // _TIMER_H
