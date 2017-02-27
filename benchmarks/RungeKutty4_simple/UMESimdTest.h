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

template<typename SCALAR_T, int PROBLEM_LENGTH, int STRIDE, int STEP_COUNT>
class UMESimdTest : public Test {
private:
    static const int OPTIMAL_ALIGNMENT = UME::VECTOR::Vector<SCALAR_T, PROBLEM_LENGTH, STRIDE>::ALIGNMENT();

    template<typename USER_LAMBDA_T >
    UME_NEVER_INLINE void rk4_vectorized(
        UME::SIMD::SIMDVec<SCALAR_T, STRIDE> & result,
        UME::SIMD::SIMDVec<SCALAR_T, STRIDE> x,
        UME::SIMD::SIMDVec<SCALAR_T, STRIDE> y,
        SCALAR_T dx,
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

    template<typename USER_LAMBDA_T >
    UME_NEVER_INLINE void rk4_scalar(
        UME::SIMD::SIMDVec<SCALAR_T, 1> & result,
        UME::SIMD::SIMDVec<SCALAR_T, 1> x,
        UME::SIMD::SIMDVec<SCALAR_T, 1> y,
        SCALAR_T dx,
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

    SCALAR_T* x;
    SCALAR_T* y;

    SCALAR_T *y_initial; // Copy of initial values of 'y' for verification

public:
    UME_NEVER_INLINE virtual void initialize() {
        // Set initial values
        std::random_device rd;
        std::mt19937 gen(rd());

        x = (SCALAR_T*)UME::DynamicMemory::AlignedMalloc(sizeof(SCALAR_T)*PROBLEM_LENGTH, OPTIMAL_ALIGNMENT);
        y = (SCALAR_T*)UME::DynamicMemory::AlignedMalloc(sizeof(SCALAR_T)*PROBLEM_LENGTH, OPTIMAL_ALIGNMENT);

        y_initial = (SCALAR_T*)UME::DynamicMemory::AlignedMalloc(sizeof(SCALAR_T)*PROBLEM_LENGTH, OPTIMAL_ALIGNMENT);

        for (int i = 0; i < PROBLEM_LENGTH; i++) {
            y_initial[i] = randomValue<SCALAR_T>(gen);
            y[i] = y_initial[i];
            x[i] = randomValue<SCALAR_T>(gen);
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code() {
        int steps = STEP_COUNT; // Number of time steps to calculate
        float timestep = 0.001f;

        auto userFunction = [](auto X, auto Y) { return X * X + Y; };

        for (int i = 0; i < STEP_COUNT; i++) {

            int LOOP_COUNT = PROBLEM_LENGTH / STRIDE;
            int LOOP_PEEL_OFFSET = LOOP_COUNT * STRIDE;

            for(int j = 0; j < LOOP_PEEL_OFFSET; j += STRIDE)
            {
                UME::SIMD::SIMDVec<SCALAR_T, STRIDE> result_vec;
                UME::SIMD::SIMDVec<SCALAR_T, STRIDE> x_vec;
                UME::SIMD::SIMDVec<SCALAR_T, STRIDE> y_vec;

                x_vec.loada(&x[j]);
                y_vec.loada(&y[j]);

                // Calculate the derivative
                rk4_vectorized(
                    result_vec,
                    x_vec,
                    y_vec,
                    timestep,
                    userFunction);

                // Update value
                result_vec.storea(&y[j]);

                // Update timestep
                x_vec = x_vec + timestep;
                x_vec.storea(&x[j]);
            }

            for (int j = LOOP_PEEL_OFFSET; j < PROBLEM_LENGTH; j++)
            {
                UME::SIMD::SIMDVec<SCALAR_T, 1> result_vec;
                UME::SIMD::SIMDVec<SCALAR_T, 1> x_vec;
                UME::SIMD::SIMDVec<SCALAR_T, 1> y_vec;

                x_vec.loada(&x[j]);
                y_vec.loada(&y[j]);

                // Calculate the derivative
                rk4_scalar(
                    result_vec,
                    x_vec,
                    y_vec,
                    timestep,
                    userFunction);

                // Update value
                result_vec.storea(&y[j]);

                // Update timestep
                x_vec = x_vec + timestep;
                x_vec.storea(&x[j]);
            }

            /*
            std::cout << "Iteration: " << i << "\n";
            for (int j = 0; j < 5; j++) {
            std::cout <<
            "    x(" << x_vec.elements[j] << ") "
            "y(" << y_vec.elements[j] << ") "
            "res(" << result_vec.elements[j] << ")\n";
            }*/

        }
    }

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(y_initial);
        UME::DynamicMemory::AlignedFree(x);
        UME::DynamicMemory::AlignedFree(y);
    }

    UME_NEVER_INLINE virtual void verify() {
        // TODO:
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";

        retval += "UME::SIMD (X*X+Y), " +
            ScalarToString<SCALAR_T>::value() + " " +
            std::to_string(PROBLEM_LENGTH) + " " +
            std::to_string(STRIDE);

        return retval;
    }
};

