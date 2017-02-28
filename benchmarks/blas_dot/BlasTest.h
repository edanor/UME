// The MIT License (MIT)
//
// Copyright (c) 2016-2017 CERN
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
#ifndef BLAS_BENCH_H_
#define BLAS_BENCH_H_

#include <assert.h>

#include <umesimd/UMESimd.h>

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"

#ifdef USE_BLAS
#include <cblas.h>

template<typename FLOAT_T>
class AXPY_kernel {
    // Only specializations of this class should be allowed.
private:
    AXPY_kernel() {}
    ~AXPY_kernel() {}
};

template<>
class AXPY_kernel<float> {
public:
    UME_FORCE_INLINE static void blas_axpy(int N, float a, float *x, float *y) {
        cblas_saxpy(N, a, x, 1, y, 1);
    }
};

template<>
class AXPY_kernel<double> {
public:
    UME_FORCE_INLINE static void blas_axpy(int N, double a, double *x, double *y) {
        cblas_daxpy(N, a, x, 1, y, 1);
    }
};

template<typename FLOAT_T>
class DOT_kernel {
    // Only specializations of this class should be allowed.
private:
    DOT_kernel() {}
    ~DOT_kernel() {}
};

template<>
class DOT_kernel<float> {
public:
    UME_FORCE_INLINE static float blas_dot(int N, float *x, float *y) {
        return cblas_sdot(N, x, 1, y, 1);
    }
};

template<>
class DOT_kernel<double> {
public:
    UME_FORCE_INLINE static double blas_dot(int N, double *x, double *y) {
        return cblas_ddot(N, x, 1, y, 1);
    }
};

template<typename FLOAT_T>
class BlasSingleTest : public Test {
    int problem_size;

    FLOAT_T *x, *y;
    FLOAT_T dot_result;

public:
    BlasSingleTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        y = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code() {
        dot_result = DOT_kernel<FLOAT_T>::blas_dot(problem_size, x, y);
    }

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(x);
        UME::DynamicMemory::AlignedFree(y);
    }

    UME_NEVER_INLINE virtual void verify() { 
        // TODO
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "BLAS single, " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(problem_size);
        return retval;
    }
};

template<typename FLOAT_T>
class BlasChainedTest : public Test {
    int problem_size;

    FLOAT_T *x0, *x1, *y0, *y1;
    FLOAT_T dot_result;

public:
    BlasChainedTest(int problem_size) : Test(true), problem_size(problem_size) {}

    // All the member functions are forced to never inline,
    // so that the compiler doesn't make any opportunistic guesses.
    // Since the cost consuming part of the benchmark, contained
    // in 'benchmarked_code' is measured all at once, the 
    // measurement offset caused by virtual function call should be
    // negligible.
    UME_NEVER_INLINE virtual void initialize()
    {
        x0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x2 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x3 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x4 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x5 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x6 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x7 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x8 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x9 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        y = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);

        srand((unsigned int)time(NULL));

        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            x0[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x1[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x2[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x3[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x4[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x5[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x6[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x7[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x8[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x9[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y[i] = 0.0f;
        }

        for (int i = 0; i < 10; i++)
        {
            alpha[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        AXPY_kernel<FLOAT_T>::blas_axpy(problem_size, alpha[0], x0, y);
        AXPY_kernel<FLOAT_T>::blas_axpy(problem_size, alpha[1], x1, y);
        AXPY_kernel<FLOAT_T>::blas_axpy(problem_size, alpha[2], x2, y);
        AXPY_kernel<FLOAT_T>::blas_axpy(problem_size, alpha[3], x3, y);
        AXPY_kernel<FLOAT_T>::blas_axpy(problem_size, alpha[4], x4, y);
        AXPY_kernel<FLOAT_T>::blas_axpy(problem_size, alpha[5], x5, y);
        AXPY_kernel<FLOAT_T>::blas_axpy(problem_size, alpha[6], x6, y);
        AXPY_kernel<FLOAT_T>::blas_axpy(problem_size, alpha[7], x7, y);
        AXPY_kernel<FLOAT_T>::blas_axpy(problem_size, alpha[8], x8, y);
        AXPY_kernel<FLOAT_T>::blas_axpy(problem_size, alpha[9], x9, y);
    }

    UME_NEVER_INLINE virtual void cleanup() {

        UME::DynamicMemory::AlignedFree(x0);
        UME::DynamicMemory::AlignedFree(x1);
        UME::DynamicMemory::AlignedFree(x2);
        UME::DynamicMemory::AlignedFree(x3);
        UME::DynamicMemory::AlignedFree(x4);
        UME::DynamicMemory::AlignedFree(x5);
        UME::DynamicMemory::AlignedFree(x6);
        UME::DynamicMemory::AlignedFree(x7);
        UME::DynamicMemory::AlignedFree(x8);
        UME::DynamicMemory::AlignedFree(x9);
        UME::DynamicMemory::AlignedFree(y);
        UME::DynamicMemory::AlignedFree(alpha);

    }

    UME_NEVER_INLINE virtual void verify() { 
        // TODO
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "BLAS chained, " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(problem_size);
        return retval;
    }
};

#else

// Blas requires external dependencies. This fallback prevents
// compile time error, allowing the user to decide whether to 
// enable BLAS interface or not.
template<typename FLOAT_T>
class BlasSingleTest : public Test {
public:
    int problem_size;

    BlasSingleTest(int problem_size) : Test(false), problem_size(problem_size) {}

    // All the member functions are forced to never inline,
    // so that the compiler doesn't make any opportunistic guesses.
    // Since the cost consuming part of the benchmark, contained
    // in 'benchmarked_code' is measured all at once, the 
    // measurement offset caused by virtual function call should be
    // negligible.
    UME_NEVER_INLINE virtual void initialize() {}
    UME_NEVER_INLINE virtual void benchmarked_code() {}
    UME_NEVER_INLINE virtual void cleanup() {}
    UME_NEVER_INLINE virtual void verify() {}
    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "BLAS single, " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(problem_size);
        return retval;
    }
};
/*
template<typename FLOAT_T>
class BlasChainedTest : public Test {
public:
    int problem_size;

    BlasChainedTest(int problem_size) : Test(false), problem_size(problem_size) {}

    // All the member functions are forced to never inline,
    // so that the compiler doesn't make any opportunistic guesses.
    // Since the cost consuming part of the benchmark, contained
    // in 'benchmarked_code' is measured all at once, the 
    // measurement offset caused by virtual function call should be
    // negligible.
    UME_NEVER_INLINE virtual void initialize() {}
    UME_NEVER_INLINE virtual void benchmarked_code() {}
    UME_NEVER_INLINE virtual void cleanup() {}
    UME_NEVER_INLINE virtual void verify() {}
    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "BLAS chained, " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(problem_size);
        return retval;
    }
};
*/
#endif

#endif
