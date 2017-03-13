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

#ifndef SCALAR_BENCH_H_
#define SCALAR_BENCH_H_

#include <umesimd/UMESimd.h>

#include "RotTest.h"

#include "../utilities/UMEScalarToString.h"

// Test single execution of naive AXPY kernel.
template<typename FLOAT_T>
class ScalarSingleTest : public RotSingleTest<FLOAT_T> {
private:
    UME_FORCE_INLINE void scalar_rot(int N, FLOAT_T* X, FLOAT_T* Y, FLOAT_T c, FLOAT_T s) {
        for (int i = 0; i < N; i++) {
            FLOAT_T t0 = c * X[i] + s * Y[i];
            FLOAT_T t1 = c * Y[i] - s * X[i];
            X[i] = t0;
            Y[i] = t1;
        }
    }

public:
    ScalarSingleTest(int problem_size) : RotSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        scalar_rot(this->problem_size, this->x, this->y, this->c, this->s);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "Scalar single";
        return retval;
    }
};
/*
// Test chained execution of naive AXPY kernel.
template<typename FLOAT_T>
class ScalarChainedTest : public Test {
private:
    static const int OPTIMAL_ALIGNMENT = 64;

    FLOAT_T *x0, *x1, *y0, *y1;
    FLOAT_T alpha0, alpha1;
    FLOAT_T dot_result;

    int problem_size;

public:
    ScalarChainedTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        y0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        y1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            x0[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x1[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y0[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y1[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }

        alpha0 = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        alpha1 = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
    }

    UME_NEVER_INLINE virtual void benchmarked_code() {
        dot_result = FLOAT_T(0.0f);
        for (int i = 0; i < problem_size; i++) {
            y0[i] = alpha0*x0[i] + y0[i];
            y1[i] = alpha1*x1[i] + y1[i];
            dot_result += y0[i] * x0[i];
        }
    }

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(x0);
        UME::DynamicMemory::AlignedFree(x1);
        UME::DynamicMemory::AlignedFree(y0);
        UME::DynamicMemory::AlignedFree(y1);
    }

    UME_NEVER_INLINE virtual void verify() {
        // TODO:
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "Scalar dot(axpy(x0, t0), axpy(x1, y1)), (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(problem_size);
        return retval;
    }
};*/

#endif