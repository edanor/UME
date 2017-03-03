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
#ifndef BLAS_DOT_BENCH_H_
#define BLAS_DOT_BENCH_H_

#include <assert.h>

#include <umesimd/UMESimd.h>

#include "DotTest.h"
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
class BlasSingleTest : public DotSingleTest<FLOAT_T> {
public:
    BlasSingleTest(int problem_size) : DotSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        this->dot_result = DOT_kernel<FLOAT_T>::blas_dot(this->problem_size, this->x, this->y);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "BLAS single, " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(this->problem_size);
        return retval;
    }
};

template<typename FLOAT_T>
class BlasChainedTest : public DotChainedTest<FLOAT_T> {
public:
    BlasChainedTest(int problem_size) : DotChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        AXPY_kernel<FLOAT_T>::blas_axpy(this->problem_size, this->alpha0, this->x0, this->y0);
        AXPY_kernel<FLOAT_T>::blas_axpy(this->problem_size, this->alpha1, this->x1, this->y1);
        this->dot_result = DOT_kernel<FLOAT_T>::blas_dot(this->problem_size, this->y0, this->y1);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "BLAS dot(axpy(x0, y0), axpy(x1, y1)), " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(this->problem_size);
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
        retval += "BLAS dot(axpy(x0, y0), axpy(x1, y1)), " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(problem_size);
        return retval;
    }
};

#endif

#endif
