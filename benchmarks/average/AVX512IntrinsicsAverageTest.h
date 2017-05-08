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

#include "AverageTest.h"

template<typename FLOAT_T>
class AVX512AverageTest : public AverageTest<FLOAT_T> {
public:
    // The default class will be called when the AVX/AVX2 features are disabled
    AVX512AverageTest(int problem_size) : AverageTest<FLOAT_T> (false, problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {}
    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "AVX/AVX2 intrinsics, " + ScalarToString<FLOAT_T>::value() + " " + std::to_string(this->problem_size);
        return retval;
    }
};

#if defined(__AVX512F__) || defined(__MIC__)
template<>
class AVX512AverageTest<float> : public AverageTest<float> {
public:
    AVX512AverageTest(int problem_size) : AverageTest<float>(true, problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        float sum = 0.0f;

        // Calculate loop-peeling division
        int PEEL_OFFSET = (this->problem_size/16)*16;

        __m512 x_vec;
        __m512 sum_vec = _mm512_setzero_ps();
        // Instead of adding single elements, we are using SIMD to add elements
        // with STRIDE-16 distance. We then perform reduction using scalar code
        for(int i = 0; i < PEEL_OFFSET; i+=16)
        {
            x_vec = _mm512_load_ps(&this->x[i]); // load elements with STRIDE-16
            sum_vec = _mm512_add_ps(sum_vec, x_vec); // accumulate sum of values
        }

        // Perform reduction
        sum = _mm512_reduce_add_ps(sum_vec);

        // Calculating loop reminder
        for(int i = PEEL_OFFSET; i < this->problem_size; i++)
        {
            sum += this->x[i];
        }

        this->calculated_average = sum/(float)this->problem_size;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        
        retval += "AVX512/KNC intrinsics, " + ScalarToString<float>::value() + " " + std::to_string(this->problem_size);
        
        return retval;
    }
};

template<>
class AVX512AverageTest<double> : public AverageTest<double> {
public:
    AVX512AverageTest(int problem_size) : AverageTest<double>(true, problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        double sum = 0.0f;

        // Calculate loop-peeling division
        int PEEL_OFFSET = (this->problem_size/8)*8;

        __m512d x_vec;
        __m512d sum_vec = _mm512_setzero_pd();
        // Instead of adding single elements, we are using SIMD to add elements
        // with STRIDE-8 distance. We then perform reduction using scalar code
        for(int i = 0; i < PEEL_OFFSET; i+=8)
        {
            x_vec = _mm512_load_pd(&this->x[i]); // load elements with STRIDE-8
            sum_vec = _mm512_add_pd(sum_vec, x_vec); // accumulate sum of values
        }

        // Perform reduction
        sum = _mm512_reduce_add_pd(sum_vec);

        // Calculating loop reminder
        for(int i = PEEL_OFFSET; i < this->problem_size; i++)
        {
            sum += this->x[i];
        }

        this->calculated_average = sum/(double)this->problem_size;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "AVX512/KNC intrinsics, " + ScalarToString<double>::value() + " " + std::to_string(this->problem_size);
        return retval;
    }
};

#endif
