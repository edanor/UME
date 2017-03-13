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

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"

#include "AxpyTest.h"

// Test single execution of naive AXPY kernel.
template<typename FLOAT_T>
class ScalarSingleTest : public AxpySingleTest<FLOAT_T> {
private:
    UME_FORCE_INLINE void scalar_axpy(int N, FLOAT_T a, FLOAT_T* X, FLOAT_T* Y) {
        for (int i = 0; i < N; i++) {
            Y[i] = a*X[i] + Y[i];
        }
    }

public:
    ScalarSingleTest(int problem_size) : AxpySingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        scalar_axpy(this->problem_size, this->alpha, this->x, this->y);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "Scalar single";
        return retval;
    }

};

// Test chained execution of naive AXPY kernel.
template<typename FLOAT_T>
class ScalarChainedTest : public AxpyChainedTest<FLOAT_T> {
private:

    UME_FORCE_INLINE void scalar_axpy(int N, FLOAT_T a, FLOAT_T* X, FLOAT_T* Y) {
        for (int i = 0; i < N; i++) {
            Y[i] = a*X[i] + Y[i];
        }
    }

public:
    ScalarChainedTest(int problem_size) : AxpyChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        scalar_axpy(this->problem_size, this->alpha[0], this->x0, this->y);
        scalar_axpy(this->problem_size, this->alpha[1], this->x1, this->y);
        scalar_axpy(this->problem_size, this->alpha[2], this->x2, this->y);
        scalar_axpy(this->problem_size, this->alpha[3], this->x3, this->y);
        scalar_axpy(this->problem_size, this->alpha[4], this->x4, this->y);
        scalar_axpy(this->problem_size, this->alpha[5], this->x5, this->y);
        scalar_axpy(this->problem_size, this->alpha[6], this->x6, this->y);
        scalar_axpy(this->problem_size, this->alpha[7], this->x7, this->y);
        scalar_axpy(this->problem_size, this->alpha[8], this->x8, this->y);
        scalar_axpy(this->problem_size, this->alpha[9], this->x9, this->y);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "Scalar chained";
        return retval;
    }

};

#endif