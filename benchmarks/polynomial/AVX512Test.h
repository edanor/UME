// The MIT License (MIT)
//
// Copyright (c) 2015-2017 CERN
//
// Author: Przemyslaw Karpinski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//
//  This piece of code was developed as part of ICE-DIP project at CERN.
//  "ICE-DIP is a European Industrial Doctorate project funded by the European Community's 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".
//
#pragma once

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"

template<typename FLOAT_T>
class AVX512Test : public Test {
private:
    int problem_size;

public:
    AVX512Test(int problem_size) : Test(false), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {}
    UME_NEVER_INLINE virtual void benchmarked_code() {}
    UME_NEVER_INLINE virtual void cleanup() {}
    UME_NEVER_INLINE virtual void verify() {}
    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "AVX512, " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(problem_size);
        return retval;
    }
};

#if defined(__AVX512F__)
template<>
class AVX512Test<float> : public Test {
private:
    int problem_size;

    float a[17];
    float *x;
    float *y;
public:
    AVX512Test(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x = (float *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(float), 64);
        y = (float *)UME::DynamicMemory::AlignedMalloc(16 * sizeof(float), 64);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < ARRAY_SIZE; i++)
        {
            // Generate random numbers in range (0.0;1000.0)
            x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }

        for (int i = 0; i < 17; i++)
        {
            // Generate random coefficients in range (0.0; 1.0)
            a[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code() {
        // The polynomial calculated is as follows:
        // y(x) = a0       + a1*x     + a2*x^2   + a3*x^3   + a4*x^4   + a5*x^5   +
        //      + a6*x^6   + a7*x^7   + a8*x^8   + a9*x^9   + a10*x^10 + a11*x^11 +
        //      + a12*x^12 + a13*x^13 + a14*x^14 + a15*x^15 + a16*x^16
        //
        // With Estrin's scheme it can be simplified:
        // y(x) = (a0 + a1*x)  + x^2*(a2  + a3*x)  + x^4*(a4  + a5*x  + x^2*(a6  + a7*x)) +
        //      + x^8*(a8+a9*x + x^2*(a10 + a11*x) + x^4*(a12 + a13*x + x^2*(a14 + a15*x)) +
        //      + a16*x^16
        //

        __m512 x_vec, x2_vec, x4_vec, x8_vec, x16_vec;
        __m512 y_vec;
        __m512 t0, t1, t2, t3;

        __m512 a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16;

        a0 = _mm512_set1_ps(a[0]);
        a1 = _mm512_set1_ps(a[1]);
        a2 = _mm512_set1_ps(a[2]);
        a3 = _mm512_set1_ps(a[3]);
        a4 = _mm512_set1_ps(a[4]);
        a5 = _mm512_set1_ps(a[5]);
        a6 = _mm512_set1_ps(a[6]);
        a7 = _mm512_set1_ps(a[7]);
        a8 = _mm512_set1_ps(a[8]);
        a9 = _mm512_set1_ps(a[9]);
        a10 = _mm512_set1_ps(a[10]);
        a11 = _mm512_set1_ps(a[11]);
        a12 = _mm512_set1_ps(a[12]);
        a13 = _mm512_set1_ps(a[13]);
        a14 = _mm512_set1_ps(a[14]);
        a15 = _mm512_set1_ps(a[15]);
        a16 = _mm512_set1_ps(a[16]);

        for (int i = 0; i < ARRAY_SIZE; i += 16) {
            //x_vec.load(&x[i]);
            x_vec = _mm512_load_ps(&x[i]);

            //x2_vec = x_vec.mul(x_vec);
            x2_vec = _mm512_mul_ps(x_vec, x_vec);
            //x4_vec = x2_vec.mul(x2_vec);
            x4_vec = _mm512_mul_ps(x2_vec, x2_vec);
            //x8_vec = x4_vec.mul(x4_vec);
            x8_vec = _mm512_mul_ps(x4_vec, x4_vec);
            //x16_vec = x8_vec.mul(x8_vec);
            x16_vec = _mm512_mul_ps(x8_vec, x8_vec);

            /*
            y_vec = (a[0] + a[1]*x_vec)
            + x2_vec*(a[2] + a[3]*x_vec)
            + x4_vec*(a[4] + a[5]*x_vec + x2_vec*(a[6] + a[7]*x_vec))
            + x8_vec*(a[8] + a[9]*x_vec + x2_vec*(a[10]+a[11]*x_vec)
            + x4_vec*(a[12] + a[13]*x_vec + x2_vec*(a[14] + a[15]*x_vec)))
            + x16_vec*a[16];

            // The polynomial can be reformulated
            y_vec = (x_vec*a[1] + a[0])
            + x2_vec*(x_vec*a[3] + a[2])
            + x4_vec*(x2_vec*(x_vec*a[7] + a[6]) + x_vec*a[5] + a[4]))
            + x8_vec*(x4_vec*(x2_vec*(x_vec*a[15] + a[14])+ x_vec*a[13] + a[12])
            + x2_vec*(x_vec*a[11] + a[10]) + x_vec*a[9] + a[8])
            + x16_vec*a[16];
            */

            //y_vec = x_vec.fmuladd(a1, a0);
            y_vec = _mm512_fmadd_ps(x_vec, a1, a0);
            //t0 = x_vec.fmuladd(a3, a2);
            t0 = _mm512_fmadd_ps(x_vec, a3, a2);
            //y_vec.adda(x2_vec.mul(t0));
            y_vec = _mm512_add_ps(y_vec, _mm512_mul_ps(x2_vec, t0));

            //t0 = x_vec.fmuladd(a7, a6);
            t0 = _mm512_fmadd_ps(x_vec, a7, a6);
            //t1 = x_vec.fmuladd(a5, a4);
            t1 = _mm512_fmadd_ps(x_vec, a5, a4);
            //t2 = x2_vec.fmuladd(t0, t1);
            t2 = _mm512_fmadd_ps(x2_vec, t0, t1);
            //y_vec.adda(x4_vec.mul(t2));
            y_vec = _mm512_add_ps(y_vec, _mm512_mul_ps(x4_vec, t2));

            //t0 = x_vec.fmuladd(a15, a14);
            t0 = _mm512_fmadd_ps(x_vec, a15, a14);
            //t1 = x_vec.fmuladd(a13, a12);
            t1 = _mm512_fmadd_ps(x_vec, a13, a12);
            //t2 = x2_vec.fmuladd(t0, t1);
            t2 = _mm512_fmadd_ps(x2_vec, t0, t1);
            //t0 = x_vec.fmuladd(a11, a10);
            t0 = _mm512_fmadd_ps(x_vec, a11, a10);
            //t1 = x_vec.fmuladd(a9, a8);
            t1 = _mm512_fmadd_ps(x_vec, a9, a8);
            //t3 = x2_vec.fmuladd(t0, t1);
            t3 = _mm512_fmadd_ps(x2_vec, t0, t1);
            //t0 = x4_vec.fmuladd(t2, t3);
            t0 = _mm512_fmadd_ps(x4_vec, t2, t3);
            //y_vec.adda(x8_vec.mul(t0));
            y_vec = _mm512_add_ps(y_vec, _mm512_mul_ps(x8_vec, t0));

            //y_vec.adda(x16_vec.mul(a[16]));
            y_vec = _mm512_add_ps(y_vec, _mm512_mul_ps(x16_vec, a16));

            //y_vec.store(&y[0]);
            _mm512_store_ps(&y[0], y_vec);
        }
    }

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(y);
        UME::DynamicMemory::AlignedFree(x);
    }

    UME_NEVER_INLINE virtual void verify() {
        // TODO:
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";

        retval += "AVX512, float " +
            std::to_string(problem_size);

        return retval;
    }
};

template<>
class AVX512Test<double> : public Test {
private:
    int problem_size;

    double a[17];
    double *x;
    double *y;

public:
    AVX512Test(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x = (double *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(double), 64);
        y = (double *)UME::DynamicMemory::AlignedMalloc(8 * sizeof(double), 64);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < ARRAY_SIZE; i++)
        {
            // Generate random numbers in range (0.0;1000.0)
            x[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        }

        for (int i = 0; i < 17; i++)
        {
            // Generate random coefficients in range (0.0; 1.0)
            a[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code() {
        // The polynomial calculated is as follows:
        // y(x) = a0       + a1*x     + a2*x^2   + a3*x^3   + a4*x^4   + a5*x^5   +
        //      + a6*x^6   + a7*x^7   + a8*x^8   + a9*x^9   + a10*x^10 + a11*x^11 +
        //      + a12*x^12 + a13*x^13 + a14*x^14 + a15*x^15 + a16*x^16
        //
        // With Estrin's scheme it can be simplified:
        // y(x) = (a0 + a1*x)  + x^2*(a2  + a3*x)  + x^4*(a4  + a5*x  + x^2*(a6  + a7*x)) +
        //      + x^8*(a8+a9*x + x^2*(a10 + a11*x) + x^4*(a12 + a13*x + x^2*(a14 + a15*x)) +
        //      + a16*x^16
        //

        __m512d x_vec, x2_vec, x4_vec, x8_vec, x16_vec;
        __m512d y_vec;
        __m512d t0, t1, t2, t3;

        __m512d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16;

        a0 = _mm512_set1_pd(a[0]);
        a1 = _mm512_set1_pd(a[1]);
        a2 = _mm512_set1_pd(a[2]);
        a3 = _mm512_set1_pd(a[3]);
        a4 = _mm512_set1_pd(a[4]);
        a5 = _mm512_set1_pd(a[5]);
        a6 = _mm512_set1_pd(a[6]);
        a7 = _mm512_set1_pd(a[7]);
        a8 = _mm512_set1_pd(a[8]);
        a9 = _mm512_set1_pd(a[9]);
        a10 = _mm512_set1_pd(a[10]);
        a11 = _mm512_set1_pd(a[11]);
        a12 = _mm512_set1_pd(a[12]);
        a13 = _mm512_set1_pd(a[13]);
        a14 = _mm512_set1_pd(a[14]);
        a15 = _mm512_set1_pd(a[15]);
        a16 = _mm512_set1_pd(a[16]);

        for (int i = 0; i < ARRAY_SIZE; i += 8) {
            //x_vec.load(&x[i]);
            x_vec = _mm512_load_pd(&x[i]);

            //x2_vec = x_vec.mul(x_vec);
            x2_vec = _mm512_mul_pd(x_vec, x_vec);
            //x4_vec = x2_vec.mul(x2_vec);
            x4_vec = _mm512_mul_pd(x2_vec, x2_vec);
            //x8_vec = x4_vec.mul(x4_vec);
            x8_vec = _mm512_mul_pd(x4_vec, x4_vec);
            //x16_vec = x8_vec.mul(x8_vec);
            x16_vec = _mm512_mul_pd(x8_vec, x8_vec);

            /*
            y_vec = (a[0] + a[1]*x_vec)
            + x2_vec*(a[2] + a[3]*x_vec)
            + x4_vec*(a[4] + a[5]*x_vec + x2_vec*(a[6] + a[7]*x_vec))
            + x8_vec*(a[8] + a[9]*x_vec + x2_vec*(a[10]+a[11]*x_vec)
            + x4_vec*(a[12] + a[13]*x_vec + x2_vec*(a[14] + a[15]*x_vec)))
            + x16_vec*a[16];

            // The polynomial can be reformulated
            y_vec = (x_vec*a[1] + a[0])
            + x2_vec*(x_vec*a[3] + a[2])
            + x4_vec*(x2_vec*(x_vec*a[7] + a[6]) + x_vec*a[5] + a[4]))
            + x8_vec*(x4_vec*(x2_vec*(x_vec*a[15] + a[14])+ x_vec*a[13] + a[12])
            + x2_vec*(x_vec*a[11] + a[10]) + x_vec*a[9] + a[8])
            + x16_vec*a[16];
            */

            //y_vec = x_vec.fmuladd(a1, a0);
            y_vec = _mm512_fmadd_pd(x_vec, a1, a0);
            //t0 = x_vec.fmuladd(a3, a2);
            t0 = _mm512_fmadd_pd(x_vec, a3, a2);
            //y_vec.adda(x2_vec.mul(t0));
            y_vec = _mm512_add_pd(y_vec, _mm512_mul_pd(x2_vec, t0));

            //t0 = x_vec.fmuladd(a7, a6);
            t0 = _mm512_fmadd_pd(x_vec, a7, a6);
            //t1 = x_vec.fmuladd(a5, a4);
            t1 = _mm512_fmadd_pd(x_vec, a5, a4);
            //t2 = x2_vec.fmuladd(t0, t1);
            t2 = _mm512_fmadd_pd(x2_vec, t0, t1);
            //y_vec.adda(x4_vec.mul(t2));
            y_vec = _mm512_add_pd(y_vec, _mm512_mul_pd(x4_vec, t2));

            //t0 = x_vec.fmuladd(a15, a14);
            t0 = _mm512_fmadd_pd(x_vec, a15, a14);
            //t1 = x_vec.fmuladd(a13, a12);
            t1 = _mm512_fmadd_pd(x_vec, a13, a12);
            //t2 = x2_vec.fmuladd(t0, t1);
            t2 = _mm512_fmadd_pd(x2_vec, t0, t1);
            //t0 = x_vec.fmuladd(a11, a10);
            t0 = _mm512_fmadd_pd(x_vec, a11, a10);
            //t1 = x_vec.fmuladd(a9, a8);
            t1 = _mm512_fmadd_pd(x_vec, a9, a8);
            //t3 = x2_vec.fmuladd(t0, t1);
            t3 = _mm512_fmadd_pd(x2_vec, t0, t1);
            //t0 = x4_vec.fmuladd(t2, t3);
            t0 = _mm512_fmadd_pd(x4_vec, t2, t3);
            //y_vec.adda(x8_vec.mul(t0));
            y_vec = _mm512_add_pd(y_vec, _mm512_mul_pd(x8_vec, t0));

            //y_vec.adda(x16_vec.mul(a[16]));
            y_vec = _mm512_add_pd(y_vec, _mm512_mul_pd(x16_vec, a16));

            //y_vec.store(&y[0]);
            _mm512_store_pd(&y[0], y_vec);
        }
    }

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(y);
        UME::DynamicMemory::AlignedFree(x);
    }

    UME_NEVER_INLINE virtual void verify() {
        // TODO:
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";

        retval += "AVX512, double " +
            std::to_string(problem_size);

        return retval;
    }
};
#endif
