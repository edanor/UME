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

#ifndef AXPY_BENCH_H_
#define AXPY_BENCH_H_

#include <umesimd/UMESimd.h>

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"

#include "../utilities/ttmath/ttmath/ttmath.h"

// Test single execution of naive AXPY kernel.
template<typename FLOAT_T>
class AxpySingleTest : public Test {
protected:
    static const int OPTIMAL_ALIGNMENT = 64;
    typedef ttmath::Big<2, 2> BigFloat;

    FLOAT_T *x, *y;

    FLOAT_T alpha;
    int problem_size;

    BigFloat *y_expected;

public:
    AxpySingleTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*problem_size, OPTIMAL_ALIGNMENT);
        y = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*problem_size, OPTIMAL_ALIGNMENT);
#if defined(ENABLE_VERIFICATION)
        y_expected = (BigFloat*)UME::DynamicMemory::AlignedMalloc(sizeof(BigFloat)*problem_size, OPTIMAL_ALIGNMENT);
#endif

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
#if defined(ENABLE_VERIFICATION)
            y_expected[i] = y[i];
#endif
        }

        //alpha = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        alpha = 1.0f;
    }

    UME_NEVER_INLINE virtual void benchmarked_code() = 0;

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(x);
        UME::DynamicMemory::AlignedFree(y);
#if defined (ENABLE_VERIFICATION)
        UME::DynamicMemory::AlignedFree(y_expected);
#endif
    }

    UME_NEVER_INLINE virtual void verify() {
#if defined(ENABLE_VERIFICATION)
        // calculate the expected values
        for (int i = 0; i < problem_size; i++)
        {
            y_expected[i] = BigFloat(alpha)*BigFloat(x[i]) + y_expected[i];
        }

        // Calculate infinity norm of error between two vectors
        BigFloat y_norm = 0;
        BigFloat approx_err_norm = 0;
        BigFloat norm = 0;
        for (int i = 0; i < problem_size; i++) {
            //y_norm = std::max(double(y_norm), double(std::abs(y[i])));
            y_norm = y_norm > ttmath::Abs(BigFloat(y[i])) ? y_norm : ttmath::Abs(BigFloat(y[i]));

            //approx_err_norm = std::abs(double(y_original[i]) - double(y[i]));
            approx_err_norm = ttmath::Abs(y_expected[i] - BigFloat(y[i]));
            norm = norm > approx_err_norm ? norm : approx_err_norm;
        }
        // Calculate final norm
        error_norm_bignum = norm / y_norm;
#endif
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() = 0;
};

// Test chained execution of naive AXPY kernel.
template<typename FLOAT_T>
class AxpyChainedTest : public Test {
protected:
    static const int OPTIMAL_ALIGNMENT = 64;
    typedef ttmath::Big<8, 8> BigFloat;

    FLOAT_T *x0, *x1, *x2, *x3, *x4, *x5, *x6, *x7, *x8, *x9, *y, *alpha;

    int problem_size;

    BigFloat *y_expected;

    UME_FORCE_INLINE void reference_axpy_kernel (int N, FLOAT_T a, FLOAT_T* X, BigFloat* Y) {
        for (int i = 0; i < N; i++) {
            Y[i] = BigFloat(a)*BigFloat(X[i]) + Y[i];
        }
    }

public:
    AxpyChainedTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x2 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x3 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x4 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x5 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x6 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x7 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x8 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x9 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        y = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        y_expected = (BigFloat *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(BigFloat), OPTIMAL_ALIGNMENT);

        alpha = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(10 * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);

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
            y[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y_expected[i] = y[i];
        }

        for (int i = 0; i < 10; i++) {
            alpha[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }
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
        UME::DynamicMemory::AlignedFree(y_expected);
    }

    UME_NEVER_INLINE virtual void verify() {
        // Calculate the expected values
        reference_axpy_kernel(problem_size, alpha[0], x0, y_expected);
        reference_axpy_kernel(problem_size, alpha[1], x1, y_expected);
        reference_axpy_kernel(problem_size, alpha[2], x2, y_expected);
        reference_axpy_kernel(problem_size, alpha[3], x3, y_expected);
        reference_axpy_kernel(problem_size, alpha[4], x4, y_expected);
        reference_axpy_kernel(problem_size, alpha[5], x5, y_expected);
        reference_axpy_kernel(problem_size, alpha[6], x6, y_expected);
        reference_axpy_kernel(problem_size, alpha[7], x7, y_expected);
        reference_axpy_kernel(problem_size, alpha[8], x8, y_expected);
        reference_axpy_kernel(problem_size, alpha[9], x9, y_expected);

        // Calculate infinity norm of error between two vectors
        BigFloat y_norm = 0;
        BigFloat approx_err_norm = 0;
        BigFloat norm = 0;
        for (int i = 0; i < problem_size; i++) {
            //y_norm = std::max(double(y_norm), double(std::abs(y[i])));
            y_norm = y_norm > ttmath::Abs(BigFloat(y[i])) ? y_norm : ttmath::Abs(BigFloat(y[i]));

            //approx_err_norm = std::abs(double(y_original[i]) - double(y[i]));
            approx_err_norm = ttmath::Abs(y_expected[i] - BigFloat(y[i]));
            norm = norm > approx_err_norm ? norm : approx_err_norm;
        }
        // Calculate final norm
        error_norm_bignum = norm / y_norm;
    }
};

#endif