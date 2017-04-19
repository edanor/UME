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

#ifndef GEMM_BENCH_H_
#define GEMM_BENCH_H_

#include <umesimd/UMESimd.h>

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"

#include "../utilities/ttmath/ttmath/ttmath.h"

template<typename FLOAT_T>
UME_NEVER_INLINE void ref_matmul(int N, FLOAT_T const *A, FLOAT_T const *B, FLOAT_T* result) {
    // Traverse rows of C
    for (int i = 0; i < N; i++) {
        // Traverse cols of C
        for (int j = 0; j < N; j++) {
            FLOAT_T prod = 0.0f;
            // Traverse row of A and column of B
            for (int k = 0; k < N; k++) {
                prod += A[i*N + k] * B[k*N + j];
            }
            result[i*N + j] = prod;
        }
    }
}

template<typename FLOAT_T>
void printMatrix(int N, FLOAT_T *mat)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << mat[i*N + j] << "\t" << std::flush;
        }
        std::cout << std::endl;
    }
}

// Test single execution of naive AXPY kernel.
template<typename FLOAT_T>
class MatmulSingleTest : public Test {
private:
    FLOAT_T *R_expected;

protected:
    static const int OPTIMAL_ALIGNMENT = 64;
    typedef ttmath::Big<2, 2> BigFloat;

    FLOAT_T *A, *B, *R, *temp0;

    int problem_size;

public:
    MatmulSingleTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        A = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        B = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        R = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        temp0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        R_expected = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            for (int j = 0; j < problem_size; j++) {
                A[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                B[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                R[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                temp0[i*problem_size + j] = 0.0f;
                R_expected[i*problem_size + j] =R[i*problem_size + j];
            }
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code() = 0;

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(A);
        UME::DynamicMemory::AlignedFree(B);
        UME::DynamicMemory::AlignedFree(R);
        UME::DynamicMemory::AlignedFree(temp0);
        UME::DynamicMemory::AlignedFree(R_expected);
    }

    UME_NEVER_INLINE virtual void verify() {
        // Calculate expected values
        ref_matmul(problem_size, A, B, R_expected);

        //std::cout << "Obtained:\n";
        //printMatrix(problem_size, R);

        //std::cout << "Expected:\n";
        //printMatrix(problem_size, R_expected);

        // Calculate infinty norm
        BigFloat max_err = 0;
        BigFloat norm = 0;

        for (int i = 0; i < problem_size; i++) {
            for (int j = 0; j < problem_size; j++) {
                // Calculate max distance
                BigFloat diff = ttmath::Abs(BigFloat(R[i*problem_size+j]) - R_expected[i*problem_size+j]);
                max_err = max_err > diff ? max_err : diff;

                // Calculate max value in expected vector
                BigFloat y_abs = ttmath::Abs(BigFloat(R_expected[i*problem_size + j]));
                norm = y_abs > norm ? y_abs : norm;
            }
        }

        error_norm_bignum = max_err / norm;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() = 0;
};

// Test chained execution of naive AXPY kernel.
template<typename FLOAT_T>
class MatmulChainedTest : public Test {
private:
    FLOAT_T *R_expected;

protected:
    static const int OPTIMAL_ALIGNMENT = 64;
    typedef ttmath::Big<32, 32> BigFloat;

    int problem_size;

    FLOAT_T *A, *B, *temp0, *C, *R;

public:
    MatmulChainedTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        A = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);
        B = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);
        temp0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);
        C = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);
        R = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);
        R_expected = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(problem_size * problem_size * sizeof(FLOAT_T), 64);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            for (int j = 0; j < problem_size; j++) {
                A[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                B[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                C[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                R[i*problem_size + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
                R_expected[i*problem_size + j] = R[i*problem_size + j];
                temp0[i*problem_size + j] = 0;
            }
        }
    }

    UME_NEVER_INLINE virtual void cleanup()
    {
        UME::DynamicMemory::AlignedFree(A);
        UME::DynamicMemory::AlignedFree(B);
        UME::DynamicMemory::AlignedFree(temp0);
        UME::DynamicMemory::AlignedFree(C);
        UME::DynamicMemory::AlignedFree(R);
        UME::DynamicMemory::AlignedFree(R_expected);
    }

    UME_NEVER_INLINE virtual void verify() {
        // Calculate expected values
        ref_matmul(problem_size, A, B, temp0);
        ref_matmul(problem_size, temp0, C, R_expected);

//        std::cout << "Expected intermediate:\n";
  //      printMatrix(problem_size, temp0);

    //    std::cout << "Obtained:\n";
      //  printMatrix(problem_size, R);

//        std::cout << "Expected:\n";
//        printMatrix(problem_size, R_expected);
       
        // Calculate infinty norm
        BigFloat max_err = 0;
        BigFloat norm = 0;

        for (int i = 0; i < problem_size; i++) {
            for (int j = 0; j < problem_size; j++) {
                // Calculate max distance
                BigFloat diff = ttmath::Abs(BigFloat(R[i*problem_size + j]) - R_expected[i*problem_size + j]);
                max_err = max_err > diff ? max_err : diff;

                // Calculate max value in expected vector
                BigFloat y_abs = ttmath::Abs(BigFloat(R_expected[i*problem_size + j]));
                norm = y_abs > norm ? y_abs : norm;
            }
        }

        error_norm_bignum = max_err / norm;
    }
};

#endif