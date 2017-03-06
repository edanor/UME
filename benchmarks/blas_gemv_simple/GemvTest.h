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

#ifndef GEMV_BENCH_H_
#define GEMV_BENCH_H_

#include <umesimd/UMESimd.h>

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"

#include "../utilities/ttmath/ttmath/ttmath.h"

// Test single execution of naive AXPY kernel.
template<typename FLOAT_T>
class GemvSingleTest : public Test {
protected:
    static const int OPTIMAL_ALIGNMENT = 64;
    typedef ttmath::Big<2, 2> BigFloat;

    FLOAT_T *A;
    FLOAT_T *x, *y;
    FLOAT_T alpha, beta;

    BigFloat *y_expected;

    int problem_size;

public:
    GemvSingleTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        A = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        x = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*problem_size, OPTIMAL_ALIGNMENT);
        y = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*problem_size, OPTIMAL_ALIGNMENT);
        y_expected = (BigFloat*)UME::DynamicMemory::AlignedMalloc(sizeof(BigFloat)*problem_size, OPTIMAL_ALIGNMENT);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            for (int j = 0; j < problem_size; j++) {
                A[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            }
            // Generate random numbers in range (0.0;1.0)
            x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y_expected[i] = y[i];
        }

        alpha = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        beta = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
    }

    UME_NEVER_INLINE virtual void benchmarked_code() = 0;

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(A);
        UME::DynamicMemory::AlignedFree(x);
        UME::DynamicMemory::AlignedFree(y);
        UME::DynamicMemory::AlignedFree(y_expected);
    }

    UME_NEVER_INLINE virtual void verify() {
        // Calculate expected values
        for (int i = 0; i < problem_size; i++) {
            int row_offset = i * problem_size;
            BigFloat prod = 0;
            for (int j = 0; j < problem_size; j++)
            {
                prod += BigFloat(A[row_offset + j]) *BigFloat(x[j]);
            }
            y_expected[i] = BigFloat(alpha) * prod + BigFloat(beta) * BigFloat(y[i]);
        }

        // Calculate infinty norm
        BigFloat max_err = 0;
        BigFloat norm = 0;

        for (int i = 0; i < problem_size; i++) {
            // Calculate max distance
            BigFloat diff = ttmath::Abs(BigFloat(y[i]) - y_expected[i]);
            max_err = max_err > diff ? max_err : diff;

            // Calculate max value in expected vector
            BigFloat y_abs = ttmath::Abs(BigFloat(y_expected[i]));
            norm = y_abs > norm ? y_abs : norm;
        }

        error_norm_bignum = max_err / norm;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() = 0;
};

// Test chained execution of naive AXPY kernel.
template<typename FLOAT_T>
class GemvChainedTest : public Test {
protected:
    static const int OPTIMAL_ALIGNMENT = 64;
    typedef ttmath::Big<32, 32> BigFloat;

    int problem_size;

    FLOAT_T *A0, *A1, *A2, *A3, *A4;
    FLOAT_T *x0, *x1, *x2, *x3, *x4, *y;
    FLOAT_T alpha[5], beta[5];

    BigFloat *y_expected;

public:
    GemvChainedTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        A0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);
        A1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);
        A2 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);
        A3 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);
        A4 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);
        x0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x2 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x3 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        x4 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        y = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), 64);
        y_expected = (BigFloat*)UME::DynamicMemory::AlignedMalloc(sizeof(BigFloat)*problem_size, 64);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            for (int j = 0; j < problem_size; j++) {
                A0[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                A1[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                A2[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                A3[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                A4[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            }
            x0[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x1[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x2[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x3[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            x4[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y_expected[i] = y[i];
        }

        for (int i = 0; i < 5; i++)
        {
            alpha[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            beta[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }
    }

    UME_NEVER_INLINE virtual void cleanup()
    {
        UME::DynamicMemory::AlignedFree(A0);
        UME::DynamicMemory::AlignedFree(A1);
        UME::DynamicMemory::AlignedFree(A2);
        UME::DynamicMemory::AlignedFree(A3);
        UME::DynamicMemory::AlignedFree(A4);
        UME::DynamicMemory::AlignedFree(x0);
        UME::DynamicMemory::AlignedFree(x1);
        UME::DynamicMemory::AlignedFree(x2);
        UME::DynamicMemory::AlignedFree(x3);
        UME::DynamicMemory::AlignedFree(x4);
        UME::DynamicMemory::AlignedFree(y);
        UME::DynamicMemory::AlignedFree(y_expected);
    }

    UME_NEVER_INLINE virtual void verify() {
        
        // Calculate expected values with extended precision.
        for (int i = 0; i < problem_size; i++) {
            //std::cout << "Verify: y[" << i << "] before: " << y_expected[i].ToDouble() << "\n";
            BigFloat prod = 0;
            for (int j = 0; j < problem_size; j++) {
                prod += BigFloat(A0[i * problem_size + j]) * BigFloat(x0[j]);
            }
            y_expected[i] = BigFloat(alpha[0]) * prod + BigFloat(beta[0]) * y_expected[i];
            //std::cout << "Verify: y[" << i << "] after: " << y_expected[i].ToDouble() << "\n";
        }
        //std::cout << "\n";
        for (int i = 0; i < problem_size; i++) {
            //std::cout << "Verify: y[" << i << "] before: " << y_expected[i].ToDouble() << "\n";
            BigFloat prod = 0;
            for (int j = 0; j < problem_size; j++) {
                prod += BigFloat(A1[i * problem_size + j]) * BigFloat(x1[j]);
            }
            y_expected[i] = BigFloat(alpha[1]) * prod + BigFloat(beta[1]) * y_expected[i];
            //std::cout << "Verify: y[" << i << "] after: " << y_expected[i].ToDouble() << "\n";
        }
        //std::cout << "\n";
        for (int i = 0; i < problem_size; i++) {
          //  std::cout << "Verify: y[" << i << "] before: " << y_expected[i].ToDouble() << "\n";
            BigFloat prod = 0;
            for (int j = 0; j < problem_size; j++) {
                prod += BigFloat(A2[i * problem_size + j]) * BigFloat(x2[j]);
            }
            y_expected[i] = BigFloat(alpha[2]) * prod + BigFloat(beta[2]) * y_expected[i];
            //std::cout << "Verify: y[" << i << "] after: " << y_expected[i].ToDouble() << "\n";
        }
        //std::cout << "\n";
        for (int i = 0; i < problem_size; i++) {
            //std::cout << "Verify: y[" << i << "] before: " << y_expected[i].ToDouble() << "\n";
            BigFloat prod = 0;
            for (int j = 0; j < problem_size; j++) {
                prod += BigFloat(A3[i * problem_size + j]) * BigFloat(x3[j]);
            }
            y_expected[i] = BigFloat(alpha[3]) * prod + BigFloat(beta[3]) * y_expected[i];
            //std::cout << "Verify: y[" << i << "] after: " << y_expected[i].ToDouble() << "\n";
        }
        //std::cout << "\n";
        for (int i = 0; i < problem_size; i++) {
            //std::cout << "Verify: y[" << i << "] before: " << y_expected[i].ToDouble() << "\n";
            BigFloat prod = 0;
            for (int j = 0; j < problem_size; j++) {
                prod += BigFloat(A4[i * problem_size + j]) * BigFloat(x4[j]);
            }
            y_expected[i] = BigFloat(alpha[4]) * prod + BigFloat(beta[4]) * y_expected[i];
            //std::cout << "Verify: y[" << i << "] after: " << y_expected[i].ToDouble() << "\n";
        }
        //std::cout << "\n";

        BigFloat max_err = 0;
        BigFloat y_norm = 0;

        for (int i = 0; i < problem_size; i++) {
            BigFloat diff = ttmath::Abs(BigFloat(y[i]) - y_expected[i]);
            max_err = max_err > diff ? max_err : diff;

            BigFloat y_abs = ttmath::Abs(BigFloat(y_expected[i]));
            y_norm = y_abs > y_norm ? y_abs : y_norm;
        }

        error_norm_bignum = max_err / y_norm;
    }
};

#endif