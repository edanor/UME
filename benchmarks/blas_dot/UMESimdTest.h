// The MIT License (MIT)
//
// Copyright (c) 2016 CERN
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

#pragma once

#include <umesimd/UMESimd.h>
#include "../utilities/MeasurementHarness.h"


template<typename FLOAT_T, int STRIDE>
class UMESimdSingleTest : public Test {
private:
    int problem_size;

    FLOAT_T *x, *y;
    FLOAT_T dot_result;

public:
    UMESimdSingleTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize()
    {
        int OPTIMAL_ALIGNMENT = UME::SIMD::SIMDVec<FLOAT_T, STRIDE>::alignment();
        x = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        y = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        int LOOP_COUNT = problem_size / STRIDE;
        int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

        UME::SIMD::SIMDVec<FLOAT_T, STRIDE> x_vec, y_vec, dot_vec(FLOAT_T(0));

        for (int i = 0; i < LOOP_PEEL_OFFSET; i+= STRIDE)
        {
            x_vec.loada(&x[i]);
            y_vec.loada(&y[i]);
            dot_vec += x_vec * y_vec;
        }

        dot_result = dot_vec.hadd();

        // Use scalar code to handle the reminder of elements.
        for (int i = LOOP_PEEL_OFFSET; i < problem_size; i++)
        {
            dot_result += x[i]*y[i];
        }
    }

    UME_NEVER_INLINE virtual void cleanup()
    {
        UME::DynamicMemory::AlignedFree(x);
        UME::DynamicMemory::AlignedFree(y);
    }

    UME_NEVER_INLINE virtual void verify()
    {
        // TODO
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "UME::SIMD single, (" +
            ScalarToString<FLOAT_T>::value() + ", " +
            std::to_string(STRIDE) + ") " +
            std::to_string(problem_size);
        return retval;
    }
};
/*
template<typename FLOAT_T, int STRIDE>
class UMESimdChainedTest : public Test {
private:
    int problem_size;

    FLOAT_T *x0, *x1, *x2, *x3, *x4, *x5, *x6, *x7, *x8, *x9, *y, *alpha;
public:
    UMESimdChainedTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize()
    {
        int OPTIMAL_ALIGNMENT = UME::SIMD::SIMDVec<FLOAT_T, STRIDE>::alignment();
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
        alpha = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(10 * sizeof(FLOAT_T), sizeof(FLOAT_T));

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
        }

        for (int i = 0; i < 10; i++) {
            alpha[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        int LOOP_COUNT = problem_size / STRIDE;
        int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

        UME::SIMD::SIMDVec<FLOAT_T, STRIDE> 
            x0_vec, x1_vec, x2_vec, x3_vec, x4_vec, x5_vec,
            x6_vec, x7_vec, x8_vec, x9_vec, y_vec;

        for (int i = 0; i < LOOP_PEEL_OFFSET; i+=STRIDE)
        {
            x0_vec.loada(&x0[i]);
            x1_vec.loada(&x1[i]);
            x2_vec.loada(&x2[i]);
            x3_vec.loada(&x3[i]);
            x4_vec.loada(&x4[i]);
            x5_vec.loada(&x5[i]);
            x6_vec.loada(&x6[i]);
            x7_vec.loada(&x7[i]);
            x8_vec.loada(&x8[i]);
            x9_vec.loada(&x9[i]);
            y_vec.loada(&y[i]);
            y_vec = y_vec + alpha[0] * x0_vec + alpha[1] * x1_vec +
                alpha[2] * x2_vec + alpha[3] * x3_vec +
                alpha[4] * x4_vec + alpha[5] * x5_vec +
                alpha[6] * x6_vec + alpha[7] * x7_vec +
                alpha[8] * x8_vec + alpha[9] * x9_vec;
            y_vec.storea(&y[i]);
        }

        // Use scalar code to handle the reminder of elements.
        for (int i = LOOP_PEEL_OFFSET; i < problem_size; i++)
        {
            y[i] += alpha[0] * x0[i] + alpha[1] * x1[i] +
                alpha[2] * x2[i] + alpha[3] * x3[i] +
                alpha[4] * x4[i] + alpha[5] * x5[i] +
                alpha[6] * x6[i] + alpha[7] * x7[i] +
                alpha[8] * x8[i] + alpha[9] * x9[i];
        }
    }

    UME_NEVER_INLINE virtual void cleanup()
    {
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
    }

    UME_NEVER_INLINE virtual void verify()
    {
        // TODO
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "UME::SIMD chained, (" +
            ScalarToString<FLOAT_T>::value() + ", " +
            std::to_string(STRIDE) + ") " +
            std::to_string(problem_size);
        return retval;
    }
};*/
