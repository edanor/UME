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

#ifndef SCALAR_DOT_BENCH_H_
#define SCALAR_DOT_BENCH_H_

#include <umesimd/UMESimd.h>

#include "DotTest.h"

#include "../utilities/UMEScalarToString.h"

// Test single execution of naive AXPY kernel.
template<typename FLOAT_T>
class ScalarSingleTest : public DotSingleTest<FLOAT_T> {
private:
    UME_FORCE_INLINE FLOAT_T scalar_dot(int N, FLOAT_T* X, FLOAT_T* Y) {
        FLOAT_T res = 0;
        for (int i = 0; i < N; i++) {
            res += X[i] * Y[i];
        }
        return res;
    }

public:
    ScalarSingleTest(int problem_size) : DotSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        this->dot_result = scalar_dot(this->problem_size, this->x, this->y);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "Scalar single";
        return retval;
    }

};

// Test chained execution of naive AXPY kernel.
template<typename FLOAT_T>
class ScalarChainedTest : public DotChainedTest<FLOAT_T> {
public:
    ScalarChainedTest(int problem_size) : DotChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        this->dot_result = FLOAT_T(0.0f);
        for (int i = 0; i < this->problem_size; i++) {
            this->y0[i] = this->alpha0*this->x0[i] + this->y0[i];
            this->y1[i] = this->alpha1*this->x1[i] + this->y1[i];
            this->dot_result += this->y0[i] * this->y1[i];
        }
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "Scalar dot(axpy(x0, t0), axpy(x1, y1))";
        return retval;
    }
};

#endif