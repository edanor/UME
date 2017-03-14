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

#include "AxpyTest.h"

template<typename FLOAT_T, int STRIDE>
class UMESimdSingleTest : public AxpySingleTest<FLOAT_T> {
public:
    UMESimdSingleTest(int problem_size) : AxpySingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        int LOOP_COUNT = this->problem_size / STRIDE;
        int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

        UME::SIMD::SIMDVec<FLOAT_T, STRIDE> x_vec, y_vec;

        for (int i = 0; i < LOOP_PEEL_OFFSET; i+= STRIDE)
        {
            x_vec.loada(&this->x[i]);
            y_vec.loada(&this->y[i]);
            y_vec = this->alpha*x_vec + y_vec;
            y_vec.storea(&this->y[i]);
        }

        // Use scalar code to handle the reminder of elements.
        for (int i = LOOP_PEEL_OFFSET; i < this->problem_size; i++)
        {
            this->y[i] += this->alpha*this->x[i];
        }
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "UME::SIMD single (SIMD " + std::to_string(STRIDE) + ")";
        return retval;
    }
};

template<typename FLOAT_T, int STRIDE>
class UMESimdChainedTest : public AxpyChainedTest<FLOAT_T> {
public:
    UMESimdChainedTest(int problem_size) : AxpyChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        int LOOP_COUNT = this->problem_size / STRIDE;
        int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

        UME::SIMD::SIMDVec<FLOAT_T, STRIDE> 
            x0_vec, x1_vec, x2_vec, x3_vec, x4_vec, x5_vec,
            x6_vec, x7_vec, x8_vec, x9_vec, y_vec;

        for (int i = 0; i < LOOP_PEEL_OFFSET; i+=STRIDE)
        {
            x0_vec.loada(&this->x0[i]);
            x1_vec.loada(&this->x1[i]);
            x2_vec.loada(&this->x2[i]);
            x3_vec.loada(&this->x3[i]);
            x4_vec.loada(&this->x4[i]);
            x5_vec.loada(&this->x5[i]);
            x6_vec.loada(&this->x6[i]);
            x7_vec.loada(&this->x7[i]);
            x8_vec.loada(&this->x8[i]);
            x9_vec.loada(&this->x9[i]);
            y_vec.loada(&this->y[i]);
            y_vec = y_vec + this->alpha[0] * x0_vec + this->alpha[1] * x1_vec +
                this->alpha[2] * x2_vec + this->alpha[3] * x3_vec +
                this->alpha[4] * x4_vec + this->alpha[5] * x5_vec +
                this->alpha[6] * x6_vec + this->alpha[7] * x7_vec +
                this->alpha[8] * x8_vec + this->alpha[9] * x9_vec;
            y_vec.storea(&this->y[i]);
        }

        // Use scalar code to handle the reminder of elements.
        for (int i = LOOP_PEEL_OFFSET; i < this->problem_size; i++)
        {
            this->y[i] += this->alpha[0] * this->x0[i] + this->alpha[1] * this->x1[i] +
                this->alpha[2] * this->x2[i] + this->alpha[3] * this->x3[i] +
                this->alpha[4] * this->x4[i] + this->alpha[5] * this->x5[i] +
                this->alpha[6] * this->x6[i] + this->alpha[7] * this->x7[i] +
                this->alpha[8] * this->x8[i] + this->alpha[9] * this->x9[i];
        }
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "UME::SIMD chained (SIMD " + std::to_string(STRIDE) + ")";
        return retval;
    }
};
