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

#ifndef ROT_BENCH_H_
#define ROT_BENCH_H_

#include <umesimd/UMESimd.h>

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"

#include "../utilities/ttmath/ttmath/ttmath.h"

// Test single execution of naive AXPY kernel.
template<typename FLOAT_T>
class RotSingleTest : public Test {
protected:
    static const int OPTIMAL_ALIGNMENT = 64;
    typedef ttmath::Big<8, 8> BigFloat;

    FLOAT_T *x, *y, c, s;

    FLOAT_T dot_result;
    int problem_size;

    BigFloat *x_expected, *y_expected;

public:
    RotSingleTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*problem_size, OPTIMAL_ALIGNMENT);
        y = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*problem_size, OPTIMAL_ALIGNMENT);
        x_expected = (BigFloat*)UME::DynamicMemory::AlignedMalloc(sizeof(BigFloat)*problem_size, OPTIMAL_ALIGNMENT);
        y_expected = (BigFloat*)UME::DynamicMemory::AlignedMalloc(sizeof(BigFloat)*problem_size, OPTIMAL_ALIGNMENT);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x_expected[i] = x[i];
            y_expected[i] = y[i];
        }

        FLOAT_T theta = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX) * FLOAT_T(6.28);
        c = std::cos(theta);
        s = std::sin(theta);
    }

    UME_NEVER_INLINE virtual void benchmarked_code() = 0;

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(x);
        UME::DynamicMemory::AlignedFree(y);
        UME::DynamicMemory::AlignedFree(x_expected);
        UME::DynamicMemory::AlignedFree(y_expected);
    }

    UME_NEVER_INLINE virtual void verify() {
        // Calculate expected values
        for (int i = 0; i < problem_size; i++) {
            BigFloat t0 = BigFloat(c) * x_expected[i] + BigFloat(s) * y_expected[i];
            BigFloat t1 = BigFloat(c) * y_expected[i] - BigFloat(s) * x_expected[i];
            x_expected[i] = t0;
            y_expected[i] = t1;
        }

        // Calcu
        BigFloat x_max_err = 0;
        BigFloat y_max_err = 0;
        BigFloat x_norm = 0;
        BigFloat y_norm = 0;

        for (int i = 0; i < problem_size; i++) {
            BigFloat x_diff = ttmath::Abs(BigFloat(x[i]) - x_expected[i]);
            BigFloat y_diff = ttmath::Abs(BigFloat(y[i]) - y_expected[i]);

            x_max_err = x_max_err > x_diff ? x_max_err : x_diff;
            y_max_err = y_max_err > y_diff ? y_max_err : y_diff;

            BigFloat x_abs = ttmath::Abs(BigFloat(x[i]));
            BigFloat y_abs = ttmath::Abs(BigFloat(y[i]));

            x_norm = x_abs > x_norm ? x_abs : x_norm;
            y_norm = y_abs > y_norm ? y_abs : y_norm;
        }

        BigFloat max_err = x_max_err > y_max_err ? x_max_err : y_max_err;
        BigFloat max_norm = x_norm > y_norm ? x_norm : y_norm;

        error_norm_bignum = max_err / max_norm;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() = 0;
};

/*
template<typename FLOAT_T>
class DotChainedTest : public Test {
protected:
    static const int OPTIMAL_ALIGNMENT = 64;
    typedef ttmath::Big<8, 8> BigFloat;

    FLOAT_T *x0, *x1, *y0, *y1;
    FLOAT_T alpha0, alpha1;
    FLOAT_T dot_result;

    int problem_size;

    BigFloat *x0_expected, *y0_expected, *x1_expected, *y1_expected;

public:
    DotChainedTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        y0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        y1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);

        x0_expected = (BigFloat *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(BigFloat), OPTIMAL_ALIGNMENT);
        y0_expected = (BigFloat *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(BigFloat), OPTIMAL_ALIGNMENT);
        x1_expected = (BigFloat *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(BigFloat), OPTIMAL_ALIGNMENT);
        y1_expected = (BigFloat *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(BigFloat), OPTIMAL_ALIGNMENT);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            x0[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x1[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y0[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y1[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);

            x0_expected[i] = x0[i];
            x1_expected[i] = x1[i];
            y0_expected[i] = y0[i];
            y1_expected[i] = y1[i];
        }

        alpha0 = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        alpha1 = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
    }

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(x0);
        UME::DynamicMemory::AlignedFree(x1);
        UME::DynamicMemory::AlignedFree(y0);
        UME::DynamicMemory::AlignedFree(y1);
        UME::DynamicMemory::AlignedFree(x0_expected);
        UME::DynamicMemory::AlignedFree(x1_expected);
        UME::DynamicMemory::AlignedFree(y0_expected);
        UME::DynamicMemory::AlignedFree(y1_expected);
    }

    UME_NEVER_INLINE virtual void verify() {
        BigFloat dot_expected = 0;
        for (int i = 0; i < problem_size; i++) {
            y0_expected[i] = BigFloat(alpha0)*x0_expected[i] + y0_expected[i];
            y1_expected[i] = BigFloat(alpha1)*x1_expected[i] + y1_expected[i];
            dot_expected += y0_expected[i] * y1_expected[i];
        }

        error_norm_bignum =
            ttmath::Abs(dot_expected - BigFloat(dot_result)) /
            ttmath::Abs(dot_expected);
    }
};*/

#endif