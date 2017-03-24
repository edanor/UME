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

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMERandomValues.h"

#include "RK4Test.h"

template<typename FLOAT_T, int STRIDE>
class UMESimdTest : public RK4Test<FLOAT_T> {
private:
    static const int OPTIMAL_ALIGNMENT = 64;

    template<typename USER_LAMBDA_T, int SIMD_STRIDE>
    UME_NEVER_INLINE void rk4_vectorized(
        UME::SIMD::SIMDVec<FLOAT_T, SIMD_STRIDE> & result,
        UME::SIMD::SIMDVec<FLOAT_T, SIMD_STRIDE> x,
        UME::SIMD::SIMDVec<FLOAT_T, SIMD_STRIDE> y,
        FLOAT_T dx,
        USER_LAMBDA_T & f)
    {
        float halfdx = dx * 0.5f;

        // Implement RK4 algorithm - very straightforward process.
        // the user function is here attached as a fragment of computation
        // graph, and it can be optimized for each 'k' independantly.
        auto k1 = dx * f(x, y);
        auto k2 = dx * f(x + halfdx, y + k1 * halfdx);
        auto k3 = dx * f(x + halfdx, y + k2 * halfdx);
        auto k4 = dx * f(x + dx, y + k3 * dx);

        // Merge into full computational graph and start evaluation.
        result = y + (1.0f / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    }

public:
    UMESimdTest(int problem_size, int step_count) : RK4Test<FLOAT_T>(problem_size, step_count) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        float timestep = 0.001f;

        auto userFunction = [](auto X, auto Y) { return X * X + Y; };

        for (int i = 0; i < this->step_count; i++) {

            int LOOP_COUNT = this->problem_size / STRIDE;
            int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

            for(int j = 0; j < LOOP_PEEL_OFFSET; j += STRIDE)
            {
                UME::SIMD::SIMDVec<FLOAT_T, STRIDE> result_vec;
                UME::SIMD::SIMDVec<FLOAT_T, STRIDE> x_vec;
                UME::SIMD::SIMDVec<FLOAT_T, STRIDE> y_vec;

                x_vec.loada(&this->x[j]);
                y_vec.loada(&this->y[j]);

                // Calculate the derivative
                rk4_vectorized<decltype(userFunction), STRIDE>(
                    result_vec,
                    x_vec,
                    y_vec,
                    timestep,
                    userFunction);

                // Update value
                result_vec.storea(&this->y[j]);

                // Update timestep
                x_vec = x_vec + timestep;
                x_vec.storea(&this->x[j]);
            }

            for (int j = LOOP_PEEL_OFFSET; j < this->problem_size; j++)
            {
                UME::SIMD::SIMDVec<FLOAT_T, 1> result_vec;
                UME::SIMD::SIMDVec<FLOAT_T, 1> x_vec;
                UME::SIMD::SIMDVec<FLOAT_T, 1> y_vec;

                x_vec.loada(&this->x[j]);
                y_vec.loada(&this->y[j]);

                // Calculate the derivative
                rk4_vectorized<decltype(userFunction), 1>(
                    result_vec,
                    x_vec,
                    y_vec,
                    timestep,
                    userFunction);

                // Update value
                result_vec.storea(&this->y[j]);

                // Update timestep
                x_vec = x_vec + timestep;
                x_vec.storea(&this->x[j]);
            }
        }
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";

        retval += "UME::SIMD (X*X+Y), " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(this->problem_size) + " " +
            std::to_string(STRIDE);

        return retval;
    }
};

