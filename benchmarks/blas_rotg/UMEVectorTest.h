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

#include <umevector/UMEVector.h>
#include "../utilities/MeasurementHarness.h"
#include <umevector/evaluators/DyadicEvaluator.h>

template<typename FLOAT_T>
class UMEVectorSingleTest : public Test {
private:
    int problem_size;

    FLOAT_T *x, *y, c, s;

public:
    UMEVectorSingleTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize()
    {
        x = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        y = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1.0)
            x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            y[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }

        FLOAT_T theta = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX) * FLOAT_T(6.28);
        c = std::cos(theta);
        s = std::sin(theta);
    }

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        UME::VECTOR::Vector<FLOAT_T> x_vec(problem_size, x);
        UME::VECTOR::Vector<FLOAT_T> y_vec(problem_size, y);

        auto t0 = c * x_vec + s * y_vec;
        auto t1 = c * y_vec - s * x_vec;

        UME::VECTOR::DyadicEvaluator eval(x_vec, t0, y_vec, t1);
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
        retval += "UME::VECTOR single, (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(problem_size);
        return retval;
    }
};

template<typename FLOAT_T>
class UMEVectorChainedTest : public Test {
private:
    int problem_size;

    FLOAT_T *x0, *x1, *y0, *y1;
    FLOAT_T alpha0, alpha1;

    FLOAT_T dot_result;

public:
    UMEVectorChainedTest(int problem_size) : Test(true), problem_size(problem_size) {}


    UME_NEVER_INLINE virtual void initialize()
    {
        x0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        x1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        y0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        y1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));

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
        UME::VECTOR::Vector<FLOAT_T> x0_vec(problem_size, x0);
        UME::VECTOR::Vector<FLOAT_T> x1_vec(problem_size, x1);
        UME::VECTOR::Vector<FLOAT_T> y0_vec(problem_size, y0);
        UME::VECTOR::Vector<FLOAT_T> y1_vec(problem_size, y1);

        auto t0 = y0_vec.adda(alpha0*x0_vec); // equivalent of Y = aX + Y (AXPY)
        auto t1 = y1_vec.adda(alpha1*x1_vec);
        dot_result = (t0 * t1).hadd();
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
        retval += "UME::VECTOR dot(axpy(x0, y0), axpy(x1, y1)), (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(problem_size);
        return retval;
    }
};