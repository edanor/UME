// The MIT License (MIT)
//
// Copyright (c) 2015-2017 CERN
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

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"


template<typename FLOAT_T, int STRIDE>
class UMESimdTest : public Test {
private:
    typedef typename UME::SIMD::SIMDVec<FLOAT_T, STRIDE> FLOAT_VEC_TYPE;

    int problem_size;

    FLOAT_T a[17];
    FLOAT_T *x;
    FLOAT_T *y;

public:
    UMESimdTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), FLOAT_VEC_TYPE::alignment());
        y = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(FLOAT_VEC_TYPE::length() * sizeof(FLOAT_T), FLOAT_VEC_TYPE::alignment());

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1000.0)
            x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }

        for (int i = 0; i < 17; i++)
        {
            // Generate random coefficients in range (0.0; 1.0)
            a[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code() {
        // The polynomial calculated is as follows:
        // y(x) = a0       + a1*x     + a2*x^2   + a3*x^3   + a4*x^4   + a5*x^5   +
        //      + a6*x^6   + a7*x^7   + a8*x^8   + a9*x^9   + a10*x^10 + a11*x^11 +
        //      + a12*x^12 + a13*x^13 + a14*x^14 + a15*x^15 + a16*x^16
        //
        // With Estrin's scheme it can be simplified:
        // y(x) = (a0 + a1*x)  + x^2*(a2  + a3*x)  + x^4*(a4  + a5*x  + x^2*(a6  + a7*x)) +
        //      + x^8*(a8+a9*x + x^2*(a10 + a11*x) + x^4*(a12 + a13*x + x^2*(a14 + a15*x)) +
        //      + a16*x^16
        //

        FLOAT_VEC_TYPE x_vec, x2_vec, x4_vec, x8_vec, x16_vec;
        FLOAT_VEC_TYPE y_vec;
        FLOAT_VEC_TYPE t0, t1, t2, t3;

        FLOAT_VEC_TYPE a0(a[0]), a1(a[1]), a2(a[2]), a3(a[3]),
            a4(a[4]), a5(a[5]), a6(a[6]), a7(a[7]),
            a8(a[8]), a9(a[9]), a10(a[10]), a11(11),
            a12(a[12]), a13(a[13]), a14(a[14]), a15(a[15]);

        for (int i = 0; i < problem_size; i += FLOAT_VEC_TYPE::length()) {
            x_vec.load(&x[i]);
            x2_vec = x_vec.mul(x_vec);
            x4_vec = x2_vec.mul(x2_vec);
            x8_vec = x4_vec.mul(x4_vec);
            x16_vec = x8_vec.mul(x8_vec);

            /*
            y_vec = (a[0] + a[1]*x_vec)
            + x2_vec*(a[2] + a[3]*x_vec)
            + x4_vec*(a[4] + a[5]*x_vec + x2_vec*(a[6] + a[7]*x_vec))
            + x8_vec*(a[8] + a[9]*x_vec + x2_vec*(a[10]+a[11]*x_vec)
            + x4_vec*(a[12] + a[13]*x_vec + x2_vec*(a[14] + a[15]*x_vec)))
            + x16_vec*a[16];

            // The polynomial can be reformulated
            y_vec = (x_vec*a[1] + a[0])
            + x2_vec*(x_vec*a[3] + a[2])
            + x4_vec*(x2_vec*(x_vec*a[7] + a[6]) + x_vec*a[5] + a[4]))
            + x8_vec*(x4_vec*(x2_vec*(x_vec*a[15] + a[14])+ x_vec*a[13] + a[12])
            + x2_vec*(x_vec*a[11] + a[10]) + x_vec*a[9] + a[8])
            + x16_vec*a[16];
            */

            y_vec = x_vec.fmuladd(a1, a0);
            t0 = x_vec.fmuladd(a3, a2);
            y_vec.adda(x2_vec.mul(t0));

            t0 = x_vec.fmuladd(a7, a6);
            t1 = x_vec.fmuladd(a5, a4);
            t2 = x2_vec.fmuladd(t0, t1);
            y_vec.adda(x4_vec.mul(t2));

            t0 = x_vec.fmuladd(a15, a14);
            t1 = x_vec.fmuladd(a13, a12);
            t2 = x2_vec.fmuladd(t0, t1);
            t0 = x_vec.fmuladd(a11, a10);
            t1 = x_vec.fmuladd(a9, a8);
            t3 = x2_vec.fmuladd(t0, t1);
            t0 = x4_vec.fmuladd(t2, t3);
            y_vec.adda(x8_vec.mul(t0));

            y_vec.adda(x16_vec.mul(a[16]));

            y_vec.store(&y[0]);
        }
    }

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(y);
        UME::DynamicMemory::AlignedFree(x);
    }

    UME_NEVER_INLINE virtual void verify() {

    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";

        retval += "UME::SIMD, " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(STRIDE) + ", " +
            std::to_string(problem_size);

        return retval;
    }
};
