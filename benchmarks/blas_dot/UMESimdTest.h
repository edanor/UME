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

#include "DotTest.h"

template<typename FLOAT_T, int STRIDE>
class UMESimdSingleTest : public DotSingleTest<FLOAT_T> {
public:
    UMESimdSingleTest(int problem_size) : DotSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        int LOOP_COUNT = this->problem_size / STRIDE;
        int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

        UME::SIMD::SIMDVec<FLOAT_T, STRIDE> x_vec, y_vec, dot_vec(FLOAT_T(0));

        for (int i = 0; i < LOOP_PEEL_OFFSET; i+= STRIDE)
        {
            x_vec.loada(&this->x[i]);
            y_vec.loada(&this->y[i]);
            dot_vec += x_vec * y_vec;
        }

        this->dot_result = dot_vec.hadd();

        // Use scalar code to handle the reminder of elements.
        for (int i = LOOP_PEEL_OFFSET; i < this->problem_size; i++)
        {
            this->dot_result += this->x[i]* this->y[i];
        }
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "UME::SIMD single, (" +
            ScalarToString<FLOAT_T>::value() + ", " +
            std::to_string(STRIDE) + ") " +
            std::to_string(this->problem_size);
        return retval;
    }
};

template<typename FLOAT_T, int STRIDE>
class UMESimdChainedTest : public DotChainedTest<FLOAT_T> {
public:
    UMESimdChainedTest(int problem_size) : DotChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        int LOOP_COUNT = this->problem_size / STRIDE;
        int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

        UME::SIMD::SIMDVec<FLOAT_T, STRIDE> 
            x0_vec, x1_vec, y0_vec, y1_vec, t0, t1, dot_vec(FLOAT_T(0.0f));

        this->dot_result = FLOAT_T(0.0f);
        for (int i = 0; i < LOOP_PEEL_OFFSET; i+=STRIDE)
        {
            x0_vec.loada(&this->x0[i]);
            x1_vec.loada(&this->x1[i]);
            y0_vec.loada(&this->y0[i]);
            y1_vec.loada(&this->y1[i]);

            t0 = this->alpha0 * x0_vec + y0_vec;
            t1 = this->alpha1 * x1_vec + y1_vec;

            t0.storea(&this->y0[i]);
            t1.storea(&this->y1[i]);

            dot_vec += t0 * t1;
        }

        this->dot_result = dot_vec.hadd();

        // Use scalar code to handle the reminder of elements.
        for (int i = LOOP_PEEL_OFFSET; i < this->problem_size; i++)
        {
            this->y0[i] = this->alpha0*this->x0[i] + this->y0[i];
            this->y1[i] = this->alpha1*this->x1[i] + this->y1[i];

            this->dot_result += this->y0[i] * this->y1[i];
        }
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "UME::SIMD dot(axpy(x0,y0), axpy(x1, y1)), (" +
            ScalarToString<FLOAT_T>::value() + ", " +
            std::to_string(STRIDE) + ") " +
            std::to_string(this->problem_size);
        return retval;
    }
};
