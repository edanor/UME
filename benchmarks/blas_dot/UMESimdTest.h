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

template<typename FLOAT_T, int STRIDE>
class UMESimdChainedTest : public Test {
private:
    int problem_size;

    FLOAT_T *x0, *x1, *y0, *y1;
    FLOAT_T alpha0, alpha1;

    FLOAT_T dot_result;
public:
    UMESimdChainedTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize()
    {
        int OPTIMAL_ALIGNMENT = UME::SIMD::SIMDVec<FLOAT_T, STRIDE>::alignment();
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

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        int LOOP_COUNT = problem_size / STRIDE;
        int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

        UME::SIMD::SIMDVec<FLOAT_T, STRIDE> 
            x0_vec, x1_vec, y0_vec, y1_vec, t0, t1, dot_vec(FLOAT_T(0.0f));

        dot_result = FLOAT_T(0.0f);
        for (int i = 0; i < LOOP_PEEL_OFFSET; i+=STRIDE)
        {
            x0_vec.loada(&x0[i]);
            x1_vec.loada(&x1[i]);
            y0_vec.loada(&y0[i]);
            y1_vec.loada(&y1[i]);

            t0 = alpha0 * x0_vec + y0_vec;
            t1 = alpha1 * x1_vec + y1_vec;

            t0.storea(&y0[i]);
            t1.storea(&y1[i]);

            dot_vec += t0 * t1;
        }

        dot_result = dot_vec.hadd();

        // Use scalar code to handle the reminder of elements.
        for (int i = LOOP_PEEL_OFFSET; i < problem_size; i++)
        {
            y0[i] = alpha0*x0[i] + y0[i];
            y1[i] = alpha1*x1[i] + y1[i];

            dot_result += y0[i] * y1[i];
        }
    }

    UME_NEVER_INLINE virtual void cleanup()
    {
        UME::DynamicMemory::AlignedFree(x0);
        UME::DynamicMemory::AlignedFree(x1);
        UME::DynamicMemory::AlignedFree(y0);
        UME::DynamicMemory::AlignedFree(y1);
    }

    UME_NEVER_INLINE virtual void verify()
    {
        // TODO
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "UME::SIMD dot(axpy(x0,y0), axpy(x1, y1)), (" +
            ScalarToString<FLOAT_T>::value() + ", " +
            std::to_string(STRIDE) + ") " +
            std::to_string(problem_size);
        return retval;
    }
};
