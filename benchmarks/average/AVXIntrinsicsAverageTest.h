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
class AVXAverageTest : public AverageTest<FLOAT_T> {
public:
    // The default class will be called when the AVX/AVX2 features are disabled
    AVXAverageTest(int problem_size) : AverageTest<FLOAT_T> (false, problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {}
    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "AVX/AVX2 intrinsics, " + ScalarToString<FLOAT_T>::value() + " " + std::to_string(this->problem_size);
        return retval;
    }
};

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template<>
class AVXAverageTest<float> : public AverageTest<float> {
public:
    AVXAverageTest(int problem_size) : AverageTest<float>(true, problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        float sum = 0.0f;

        // Calculate loop-peeling division
        int PEEL_OFFSET = (this->problem_size/8)*8;

        __m256 x_vec;
        __m256 sum_vec = _mm256_setzero_ps();
        // Instead of adding single elements, we are using SIMD to add elements
        // with STRIDE-8 distance. We then perform reduction using scalar code
        for(int i = 0; i < PEEL_OFFSET; i+=8)
        {
            x_vec = _mm256_load_ps(&this->x[i]); // load elements with STRIDE-8
            sum_vec = _mm256_add_ps(sum_vec, x_vec); // accumulate sum of values
        }

        // Perform reduction
        __m256 t0 = _mm256_hadd_ps(sum_vec, sum_vec);
        __m256 t1 = _mm256_hadd_ps(t0, t0);
        __m128 t2 = _mm256_extractf128_ps(t1, 1);
        __m128 t3 = _mm256_castps256_ps128(t1);
        __m128 t4 = _mm_add_ps(t2, t3);
        sum = _mm_cvtss_f32(t4);

        // Calculating loop reminder
        for(int i = PEEL_OFFSET; i < this->problem_size; i++)
        {
            sum += this->x[i];
        }

        this->calculated_average = sum/(float)this->problem_size;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        
        retval += "AVX/AVX2 intrinsics, " + ScalarToString<float>::value() + " " + std::to_string(this->problem_size);
        
        return retval;
    }
};

template<>
class AVXAverageTest<double> : public AverageTest<double> {
public:
    AVXAverageTest(int problem_size) : AverageTest<double>(true, problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        alignas(32) double temp[4];
        __m256d x_vec;
        __m256d sum_vec = _mm256_setzero_pd();
        double sum = 0.0;
        
        // Calculate loop-peeling division
        int PEEL_OFFSET = (this->problem_size/4)*4;
        
        // Instead of adding single elements, we are using SIMD to add elements
        // with STRIDE-4 distance. We then perform reduction using scalar code
        for(int i = 0; i < PEEL_OFFSET; i+=4)
        {
            x_vec = _mm256_load_pd(&this->x[i]); // load elements with STRIDE-4
            sum_vec = _mm256_add_pd(sum_vec, x_vec); // accumulate sum of values
        }

        // Now the reduction operation converting a vector into a scalar value
        _mm256_store_pd(temp, sum_vec);
        for(int i = 0; i < 4; ++i)
        {
            sum += temp[i];  
        }

        // Calculating loop reminder
        for(int i = PEEL_OFFSET; i < this->problem_size; i++)
        {
            sum += this->x[i];
        }

        this->calculated_average = sum/(double)this->problem_size;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "AVX/AVX2 intrinsics, " + ScalarToString<double>::value() + " " + std::to_string(this->problem_size);
        return retval;
    }
};

#endif