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

#ifndef SCALAR_AXPY_BENCH_H_
#define SCALAR_AXPY_BENCH_H_

#include <umesimd/UMESimd.h>

#include "MatmulTest.h"

#include "../utilities/UMEScalarToString.h"

// This version doesn't handle possible aliasing
template<typename FLOAT_T>
UME_NEVER_INLINE void scalar_matmul(int N, FLOAT_T const *A, FLOAT_T const *B, FLOAT_T *C) {
    // Traverse rows of C
    for (int i = 0; i < N; i++) {
        // Traverse cols of C
        for (int j = 0; j < N; j++) {
            FLOAT_T prod = 0.0f;
            // Traverse row of A and column of B
            for (int k = 0; k < N; k++) {
                prod += A[i*N + k] * B[k*N + j];
            }

            C[i*N + j] = prod;
        }
    }
}

// Test single execution of naive AXPY kernel.
template<typename FLOAT_T>
class ScalarSingleTest : public MatmulSingleTest<FLOAT_T> {
public:
    ScalarSingleTest(int problem_size) : MatmulSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        scalar_matmul(this->problem_size, this->A, this->B, this->R);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "Scalar single";
        return retval;
    }

};

// Test chained execution of naive AXPY kernel.
template<typename FLOAT_T>
class ScalarChainedTest : public MatmulChainedTest<FLOAT_T> {
public:
    ScalarChainedTest(int problem_size) : MatmulChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        scalar_matmul(this->problem_size, this->A, this->B, this->temp0);
        scalar_matmul(this->problem_size, this->temp0, this->C, this->R);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "Scalar chained";
        return retval;
    }

};

#endif