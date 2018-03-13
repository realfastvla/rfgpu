#ifndef _IMAGE_H
#define _IMAGE_H

#include <string>
#include <vector>
#include <map>

#include <cuda.h>
#include <cuComplex.h>
#include <cufft.h>

#include "array.h"
#include "timer.h"

namespace rfgpu {

    typedef cuComplex cdata; // complex data type
    typedef cufftReal rdata; // real data type

    class ImageStatistic;

    class Image
    {
        public:
            Image(int xpix, int ypix);
            ~Image();

            void operate(Array<cdata,true> &vis, Array<rdata,true> &img);
            void operate(cdata *vis, rdata *img);

            void add_stat(std::string name);
            std::vector<std::string> stat_names() const;
            std::vector<double> stats(Array<rdata,true> &img);

            std::map<std::string,Timer *> timers;

        protected:
            void setup();

            int vispix() const { return ((ypix/2)+1) * xpix; }
            int imgpix() const { return xpix*ypix; }

            int xpix; // Number of image pixels
            int ypix; // Numer of image pixels

            cufftHandle plan;

            Array<double,true> _stats;

            std::vector<ImageStatistic *> stat_funcs;
    };

    class ImageStatistic
    {
        friend class Image;

        public:
            ImageStatistic(int xpix, int ypix, std::string name);
            ~ImageStatistic() {};

            virtual void operate(Array<rdata,true> &img, double *result) = 0;
            virtual double finalize(double result) const { return result; }

        protected:
            virtual void setup();
            virtual void calc_buffer_size() = 0;

            int xpix; 
            int ypix;

            std::string name;

            size_t tmp_bytes;  // Size of temp space on GPU
            Array<char> tmp;   // Temp buffer on GPU
    };

    class ImageMax: public ImageStatistic
    {
        public:
            ImageMax(int xpix, int ypix, std::string name="max") 
                : ImageStatistic(xpix, ypix, name) {}
            void operate(Array<rdata,true> &img, double *result);
        protected:
            void calc_buffer_size();
    };

    class ImageRMS: public ImageStatistic
    {
        public:
            ImageRMS(int xpix, int ypix, std::string name="rms") 
                : ImageStatistic(xpix, ypix, name) {}
            void operate(Array<rdata,true> &img, double *result);
            double finalize(double result) const 
                { return sqrt(result/(double)(xpix*ypix)); }
        protected:
            void calc_buffer_size();
    };

    class ImageIQR: public ImageStatistic
    {
        public:
            ImageIQR(int xpix, int ypix, std::string name="iqr") 
                : ImageStatistic(xpix, ypix, name) {}
            void operate(Array<rdata,true> &img, double *result);
        protected:
            void calc_buffer_size();
    };

}
#endif
