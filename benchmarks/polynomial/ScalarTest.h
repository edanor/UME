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


template<typename FLOAT_T>
class ScalarTest : public Test {
private:
    int problem_size;

    FLOAT_T a[17];
    FLOAT_T *x;

public:
    ScalarTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(FLOAT_T), sizeof(FLOAT_T));

        srand((unsigned int)time(NULL));
        // Initialize arrays with random data
        for (int i = 0; i < ARRAY_SIZE; i++)
        {
            // Generate random numbers in range (0.0;1.0)
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
        FLOAT_T x2, x4, x8, x16;
        volatile FLOAT_T y;

        for (int i = 0; i < ARRAY_SIZE; i++) {
            x2 = x[i] * x[i];
            x4 = x2*x2;
            x8 = x4*x4;
            x16 = x8*x8;

            y = (a[0] + a[1] * x[i])
                + x2*(a[2] + a[3] * x[i])
                + x4*(a[4] + a[5] * x[i] + x2*(a[6] + a[7] * x[i]))
                + x8*(a[8] + a[9] * x[i] + x2*(a[10] + a[11] * x[i]) + x4*(a[12] + a[13] * x[i] + x2*(a[14] + a[15] * x[i])))
                + x16*a[16];
        }

        (void)y; // avoid 'unused-variable' warnings
    }
    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(x);
    }
    UME_NEVER_INLINE virtual void verify() {

    }
    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";

        retval += "Scalar, " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(problem_size);

        return retval;
    }
};