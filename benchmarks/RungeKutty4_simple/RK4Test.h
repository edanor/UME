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

#ifndef RK4_BENCH_H_
#define RK4_BENCH_H_

#include <umesimd/UMESimd.h>

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"

#include "../utilities/ttmath/ttmath/ttmath.h"

// Test single execution of naive AXPY kernel.
template<typename FLOAT_T>
class RK4Test : public Test {
protected:
    static const int OPTIMAL_ALIGNMENT = 64;
    typedef ttmath::Big<2, 2> BigFloat;

    FLOAT_T* x;
    FLOAT_T* y;

    FLOAT_T *y_initial; // Copy of initial values of 'y' for verification

    int problem_size;
    int step_count;

    BigFloat *y_expected;

public:
    RK4Test(int problem_size, int step_count) : Test(true), problem_size(problem_size), step_count(step_count) {}

    UME_NEVER_INLINE virtual void initialize() {
        // Set initial values
        std::random_device rd;
        std::mt19937 gen(rd());

        x = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*problem_size, OPTIMAL_ALIGNMENT);
        y = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*problem_size, OPTIMAL_ALIGNMENT);
        y_initial = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*problem_size, OPTIMAL_ALIGNMENT);

        for (int i = 0; i < problem_size; i++) {
            x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y_initial[i] = y[i];
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code() = 0;

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(y_initial);
        UME::DynamicMemory::AlignedFree(x);
        UME::DynamicMemory::AlignedFree(y);
    }

    UME_NEVER_INLINE virtual void verify() {
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() = 0;
};

#endif