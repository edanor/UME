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


template<typename FLOAT_T>
class UMEVectorSingleTest : public Test {
private:
    int problem_size;

    FLOAT_T *x, *y;
    FLOAT_T alpha;
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

        alpha = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
    }

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        UME::VECTOR::Vector<FLOAT_T> x_vec(problem_size, x);
        UME::VECTOR::Vector<FLOAT_T> y_vec(problem_size, y);
        UME::VECTOR::Scalar<FLOAT_T> alpha_s(alpha);

        y_vec = alpha_s*x_vec + y_vec;
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

    FLOAT_T *x0, *x1, *x2, *x3, *x4, *x5, *x6, *x7, *x8, *x9, *y, *alpha;
public:
    UMEVectorChainedTest(int problem_size) : Test(true), problem_size(problem_size) {}


    UME_NEVER_INLINE virtual void initialize()
    {
        x0 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        x1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        x2 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        x3 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        x4 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        x5 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        x6 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        x7 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        x8 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        x9 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
        y = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), sizeof(FLOAT_T));
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
        UME::VECTOR::Vector<FLOAT_T> x0_vec(problem_size, x0);
        UME::VECTOR::Vector<FLOAT_T> x1_vec(problem_size, x1);
        UME::VECTOR::Vector<FLOAT_T> x2_vec(problem_size, x2);
        UME::VECTOR::Vector<FLOAT_T> x3_vec(problem_size, x3);
        UME::VECTOR::Vector<FLOAT_T> x4_vec(problem_size, x4);
        UME::VECTOR::Vector<FLOAT_T> x5_vec(problem_size, x5);
        UME::VECTOR::Vector<FLOAT_T> x6_vec(problem_size, x6);
        UME::VECTOR::Vector<FLOAT_T> x7_vec(problem_size, x7);
        UME::VECTOR::Vector<FLOAT_T> x8_vec(problem_size, x8);
        UME::VECTOR::Vector<FLOAT_T> x9_vec(problem_size, x9);
        UME::VECTOR::Vector<FLOAT_T> y_vec(problem_size, y);

        UME::VECTOR::Scalar<FLOAT_T> alpha0(alpha[0]);
        UME::VECTOR::Scalar<FLOAT_T> alpha1(alpha[1]);
        UME::VECTOR::Scalar<FLOAT_T> alpha2(alpha[2]);
        UME::VECTOR::Scalar<FLOAT_T> alpha3(alpha[3]);
        UME::VECTOR::Scalar<FLOAT_T> alpha4(alpha[4]);
        UME::VECTOR::Scalar<FLOAT_T> alpha5(alpha[5]);
        UME::VECTOR::Scalar<FLOAT_T> alpha6(alpha[6]);
        UME::VECTOR::Scalar<FLOAT_T> alpha7(alpha[7]);
        UME::VECTOR::Scalar<FLOAT_T> alpha8(alpha[8]);
        UME::VECTOR::Scalar<FLOAT_T> alpha9(alpha[9]);

        y_vec = y_vec + alpha0*x0_vec + alpha1*x1_vec +
            alpha2*x2_vec + alpha3*x3_vec +
            alpha4*x4_vec + alpha5*x5_vec +
            alpha6*x6_vec + alpha7*x7_vec +
            alpha8*x8_vec + alpha9*x9_vec;
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
        retval += "UME::VECTOR chained, (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(problem_size);
        return retval;
    }
};
