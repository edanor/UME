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

#include "MatmulTest.h"
#include "../utilities/UMEScalarToString.h"

#ifdef USE_BLAS

#include "BlasWrapper.h"

template<typename FLOAT_T>
class BlasSingleTest : public MatmulSingleTest<FLOAT_T> {
public:
    BlasSingleTest(int problem_size) : MatmulSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size, FLOAT_T(1.0f), this->A, this->B, FLOAT_T(0.0f), this->R);
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
class BlasChainedTest : public MatmulChainedTest<FLOAT_T> {
public:
    BlasChainedTest(int problem_size) : MatmulChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        // R = A * B
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size, FLOAT_T(1.0f), this->A, this->B, FLOAT_T(0.0f), this->temp0);
        //std::cout << "Obtained intermediate:\n";
        //printMatrix(this->problem_size, this->R);
        // R = (A*B) *C
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size, FLOAT_T(1.0f), this->temp0, this->C, FLOAT_T(0.0f), this->R);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "BLAS chained, " +
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
class BlasSingleTest : public MatmulSingleTest<FLOAT_T> {
public:
    BlasSingleTest(int problem_size) : MatmulSingleTest<FLOAT_T>(problem_size) {}

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
        std::string retval = "BLAS single";
        return retval;
    }
};

template<typename FLOAT_T>
class BlasChainedTest : public MatmulChainedTest<FLOAT_T> {
public:
    BlasChainedTest(int problem_size) : MatmulChainedTest<FLOAT_T>(problem_size) {}

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
        std::string retval = "BLAS chained";
        return retval;
    }
};

#endif

#endif
