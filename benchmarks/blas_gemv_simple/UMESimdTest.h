// The MIT License (MIT)
//
// Copyright (c) 2016-2017 CERN
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

#include "GemvTest.h"

template<typename FLOAT_T, int STRIDE>
class UMESimdSingleTest : public GemvSingleTest<FLOAT_T> {
public:
    UMESimdSingleTest(int problem_size) : GemvSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        int LOOP_COUNT = this->problem_size / STRIDE;
        int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

        UME::SIMD::SIMDVec<FLOAT_T, STRIDE> x_vec, a_vec, reduction_vec;

        // Traverse rows of A
        for (int i = 0; i < this->problem_size; i++)
        {
            int row_offset = i * this->problem_size;
            // Traverse columns of A
            reduction_vec = FLOAT_T(0.0f);
            for (int j = 0; j < LOOP_PEEL_OFFSET; j += STRIDE)
            {
                // Cannot load aligned because there is no guarantee that
                // all rows are aligned to the OPTIMAL_ALIGNMENT boundaries.
                x_vec.load(&this->x[j]);
                a_vec.load(&this->A[row_offset + j]);

                reduction_vec += a_vec * x_vec;
            }

            // Compute the SIMD part of dot product
            FLOAT_T reduction = reduction_vec.hadd();

            // Use scalar code to handle the reminder of elements.
            for (int j = LOOP_PEEL_OFFSET; j < this->problem_size; j++)
            {
                reduction += this->A[row_offset + j] * this->x[j];
            }

            this->y[i] = this->alpha * reduction + this->beta * this->y[i];
        }
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "UME::SIMD single";
        return retval;
    }
};

template<typename FLOAT_T, int STRIDE>
class UMESimdChainedTest : public GemvChainedTest<FLOAT_T> {
private:
    UME_FORCE_INLINE void simd_gemv(int N, FLOAT_T *A, FLOAT_T alpha, FLOAT_T* x, FLOAT_T beta, FLOAT_T* y) {

        int LOOP_COUNT = N / STRIDE;
        int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

        UME::SIMD::SIMDVec<FLOAT_T, STRIDE> x_vec, a_vec, reduction_vec;

        // Traverse rows of A
        for (int i = 0; i < N; i++)
        {
            int row_offset = i * N;
            // Traverse columns of A
            reduction_vec = FLOAT_T(0.0f);
            for (int j = 0; j < LOOP_PEEL_OFFSET; j += STRIDE)
            {
                // Cannot load aligned because there is no guarantee that
                // all rows are aligned to the OPTIMAL_ALIGNMENT boundaries.
                x_vec.load(&x[j]);
                a_vec.load(&A[row_offset + j]);

                reduction_vec += a_vec * x_vec;
            }

            // Compute the SIMD part of dot product
            FLOAT_T reduction = reduction_vec.hadd();


            // Use scalar code to handle the reminder of elements.
            for (int j = LOOP_PEEL_OFFSET; j < N; j++)
            {
                reduction += A[row_offset + j] * x[j];
            }

            y[i] = alpha * reduction + beta * y[i];
        }
    }

public:
    UMESimdChainedTest(int problem_size) : GemvChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        int LOOP_COUNT = this->problem_size / STRIDE;
        int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;
        // Traverse 'y'
        for (int i = 0; i < this->problem_size; i++)
        {
            int row_offset = i * this->problem_size;
            UME::SIMD::SIMDVec<FLOAT_T, STRIDE> x_vec, a_vec, reduction_vec;
            FLOAT_T reduction;
            // A0
UME::SIMD::SIMDVec<FLOAT_T, STRIDE>::prefetch0(&this->A1[row_offset]);

            reduction_vec = FLOAT_T(0.0f);
            for (int j = 0; j < LOOP_PEEL_OFFSET; j += STRIDE)
            {
                // Cannot load aligned because there is no guarantee that
                // all rows are aligned to the OPTIMAL_ALIGNMENT boundaries.
                x_vec.load(&this->x0[j]);
                a_vec.load(&this->A0[row_offset + j]);

                reduction_vec += a_vec * x_vec;
            }

            reduction = reduction_vec.hadd();

            // Use scalar code to handle the reminder of elements.
            for (int j = LOOP_PEEL_OFFSET; j < this->problem_size; j++)
            {
                reduction += this->A0[row_offset + j] * this->x0[j];
            }

            FLOAT_T t0 = this->alpha[0] * reduction + this->beta[0] * this->y[i];

            // A1
            reduction_vec = FLOAT_T(0.0f);
UME::SIMD::SIMDVec<FLOAT_T, STRIDE>::prefetch0(&this->A2[row_offset]);
            for (int j = 0; j < LOOP_PEEL_OFFSET; j += STRIDE)
            {
                // Cannot load aligned because there is no guarantee that
                // all rows are aligned to the OPTIMAL_ALIGNMENT boundaries.
                x_vec.load(&this->x1[j]);
                a_vec.load(&this->A1[row_offset + j]);

                reduction_vec += a_vec * x_vec;
            }

            reduction = reduction_vec.hadd();

            // Use scalar code to handle the reminder of elements.
            for (int j = LOOP_PEEL_OFFSET; j < this->problem_size; j++)
            {
                reduction += this->A1[row_offset + j] * this->x1[j];
            }

            FLOAT_T t1 = this->alpha[1] * reduction + this->beta[1] * t0;

            // A2
            reduction_vec = FLOAT_T(0.0f);
UME::SIMD::SIMDVec<FLOAT_T, STRIDE>::prefetch0(&this->A3[row_offset]);
            for (int j = 0; j < LOOP_PEEL_OFFSET; j += STRIDE)
            {
                // Cannot load aligned because there is no guarantee that
                // all rows are aligned to the OPTIMAL_ALIGNMENT boundaries.
                x_vec.load(&this->x2[j]);
                a_vec.load(&this->A2[row_offset + j]);

                reduction_vec += a_vec * x_vec;
            }

            reduction = reduction_vec.hadd();

            // Use scalar code to handle the reminder of elements.
            for (int j = LOOP_PEEL_OFFSET; j < this->problem_size; j++)
            {
                reduction += this->A2[row_offset + j] * this->x2[j];
            }

            FLOAT_T t2 = this->alpha[2] * reduction + this->beta[2] * t1;

            // A3
            reduction_vec = FLOAT_T(0.0f);
UME::SIMD::SIMDVec<FLOAT_T, STRIDE>::prefetch0(&this->A4[row_offset]);
            for (int j = 0; j < LOOP_PEEL_OFFSET; j += STRIDE)
            {
                // Cannot load aligned because there is no guarantee that
                // all rows are aligned to the OPTIMAL_ALIGNMENT boundaries.
                x_vec.load(&this->x3[j]);
                a_vec.load(&this->A3[row_offset + j]);

                reduction_vec += a_vec * x_vec;
            }

            reduction = reduction_vec.hadd();

            // Use scalar code to handle the reminder of elements.
            for (int j = LOOP_PEEL_OFFSET; j < this->problem_size; j++)
            {
                reduction += this->A3[row_offset + j] * this->x3[j];
            }

            FLOAT_T t3 = this->alpha[3] * reduction + this->beta[3] * t2;

            // A4
            reduction_vec = FLOAT_T(0.0f);
UME::SIMD::SIMDVec<FLOAT_T, STRIDE>::prefetch0(&this->A0[row_offset + this->problem_size]);
            for (int j = 0; j < LOOP_PEEL_OFFSET; j += STRIDE)
            {
                // Cannot load aligned because there is no guarantee that
                // all rows are aligned to the OPTIMAL_ALIGNMENT boundaries.
                x_vec.load(&this->x4[j]);
                a_vec.load(&this->A4[row_offset + j]);

                reduction_vec += a_vec * x_vec;
            }

            reduction = reduction_vec.hadd();

            // Use scalar code to handle the reminder of elements.
            for (int j = LOOP_PEEL_OFFSET; j < this->problem_size; j++)
            {
                reduction += this->A4[row_offset + j] * this->x4[j];
            }

            this->y[i] = this->alpha[4] * reduction + this->beta[4] * t3;
        }
        /*
        simd_gemv(this->problem_size, this->A0, this->alpha[0], this->x0, this->beta[0], this->y);
        simd_gemv(this->problem_size, this->A1, this->alpha[1], this->x1, this->beta[1], this->y);
        simd_gemv(this->problem_size, this->A2, this->alpha[2], this->x2, this->beta[2], this->y);
        simd_gemv(this->problem_size, this->A3, this->alpha[3], this->x3, this->beta[3], this->y);
        simd_gemv(this->problem_size, this->A4, this->alpha[4], this->x4, this->beta[4], this->y);*/
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "UME::SIMD chained";
        return retval;
    }
};
