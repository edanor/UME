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

#include "../utilities/ttmath/ttmath/ttmath.h"

template<typename FLOAT_T>
class AverageTest : public Test {
protected:
    typedef ttmath::Big<2, 2> BigFloat;
    int problem_size;

    FLOAT_T *x;     // The input data.    
    FLOAT_T calculated_average; // The average should be stored here.
public:
    AverageTest(bool test_enabled, int problem_size) : Test(test_enabled), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
        x = (FLOAT_T *) UME::DynamicMemory::AlignedMalloc(problem_size*sizeof(FLOAT_T), 64);

        // Initialize arrays with random data
        for(int i = 0; i < problem_size; i++)
        {
            // Generate random numbers in range (0.0;1000.0)
            x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX/1000);
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code() = 0;

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(x);
    }

    UME_NEVER_INLINE virtual void verify() {
        BigFloat sum = 0.0f;
        BigFloat avg = 0.0f;

        for(int i = 0; i < problem_size; i++)
        {
            sum += x[i];
        }

        avg = sum/(FLOAT_T)problem_size;

        error_norm_bignum = ttmath::Abs(avg - BigFloat(calculated_average));
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() = 0;
};
