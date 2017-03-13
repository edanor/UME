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

#include "GemvTest.h"

#include "../utilities/UMEScalarToString.h"

// Test single execution of naive AXPY kernel.
template<typename FLOAT_T>
class ScalarSingleTest : public GemvSingleTest<FLOAT_T> {
private:
    static const int OPTIMAL_ALIGNMENT = 64;

    UME_FORCE_INLINE void scalar_gemv(int N, FLOAT_T *A, FLOAT_T alpha, FLOAT_T* x, FLOAT_T beta, FLOAT_T* y) {
        // Traverse rows of A
        for (int i = 0; i < N; i++) {
            // Traverse columns of A
            int row_offset = i * N;
            FLOAT_T prod = 0;
            for (int j = 0; j < N; j++)
            {
                prod += A[row_offset + j] * x[j];
            }
            y[i] = alpha * prod + beta * y[i];
        }
    }

public:
    ScalarSingleTest(int problem_size) : GemvSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        scalar_gemv(this->problem_size, this->A, this->alpha, this->x, this->beta, this->y);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "Scalar single";
        return retval;
    }

};

// Test chained execution of naive AXPY kernel.
template<typename FLOAT_T>
class ScalarChainedTest : public GemvChainedTest<FLOAT_T> {
private:
    UME_FORCE_INLINE void scalar_gemv(int N, FLOAT_T *A, FLOAT_T alpha, FLOAT_T* x, FLOAT_T beta, FLOAT_T* y) {
        // Traverse rows of A
        for (int i = 0; i < N; i++) {
            // Traverse columns of A
            int row_offset = i * N;
            FLOAT_T t0 = 0;
            for (int j = 0; j < N; j++)
            {
                t0 += A[row_offset + j] * x[j];
            }
            //std::cout << "Scalar: y[" << i << "] before: " << y[i] << "\n";
            y[i] = alpha * t0 + beta*y[i];
            //std::cout << "Scalar: y[" << i << "] after: " << y[i] << "\n";
        }
        //std::cout << "\n";
    }

public:
    ScalarChainedTest(int problem_size) : GemvChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        scalar_gemv(this->problem_size, this->A0, this->alpha[0], this->x0, this->beta[0], this->y);
        scalar_gemv(this->problem_size, this->A1, this->alpha[1], this->x1, this->beta[1], this->y);
        scalar_gemv(this->problem_size, this->A2, this->alpha[2], this->x2, this->beta[2], this->y);
        scalar_gemv(this->problem_size, this->A3, this->alpha[3], this->x3, this->beta[3], this->y);
        scalar_gemv(this->problem_size, this->A4, this->alpha[4], this->x4, this->beta[4], this->y);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "Scalar chained";
        return retval;
    }

};

#endif