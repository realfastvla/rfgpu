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

            IFTIMER( std::map<std::string,Timer *> timers; )

        protected:
            void setup();

            int vispix() const { return ((ypix/2)+1) * xpix; }
            int imgpix() const { return xpix*ypix; }

            int xpix; // Number of image pixels
            int ypix; // Numer of image pixels

            cufftHandle plan;

            Array<double,true> _stats;
            std::vector<size_t> _stat_offs;

            std::vector<ImageStatistic *> stat_funcs;
    };

    class ImageStatistic
    {
        friend class Image;

        public:
            ImageStatistic(int xpix, int ypix, std::string name);
            ~ImageStatistic() {};

            virtual void operate(Array<rdata,true> &img, double *result) = 0;
            virtual void finalize(double *result) const {}

            // The provides function should return a list of strings giving
            // the statistics computed by the class.  This allows computations
            // which result in multiple return values.  By default a single
            // value with same name as the instance name is assumed.
            virtual std::vector<std::string> provides() const {
                return std::vector<std::string>(1, name);
            }

            // Shortcut to get the number of values returned
            int num_stats() const { return provides().size(); }

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

    class ImageMaxPixel: public ImageStatistic
    {
        public:
            ImageMaxPixel(int xpix, int ypix, std::string name="pix") 
                : ImageStatistic(xpix, ypix, name) {}
            void operate(Array<rdata,true> &img, double *result);
            std::vector<std::string> provides() const {
                std::vector<std::string> r;
                r.push_back("max");
                r.push_back("xpeak");
                r.push_back("ypeak");
                return r;
            }
            void finalize(double *result) const;
        protected:
            void calc_buffer_size();
    };

    class ImageRMS: public ImageStatistic
    {
        public:
            ImageRMS(int xpix, int ypix, std::string name="rms") 
                : ImageStatistic(xpix, ypix, name) {}
            void operate(Array<rdata,true> &img, double *result);
            void finalize(double *result) const 
                { *result = sqrt((*result)/(double)(xpix*ypix)); }
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
