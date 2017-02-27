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
#include "../utilities/UMERandomValues.h"
#include "../utilities/UMEScalarToString.h"

template<typename SCALAR_T, int PROBLEM_LENGTH, int STEP_COUNT>
class ScalarTest : public Test {
private:
    static const int OPTIMAL_ALIGNMENT = 64;
    // Scalar RK4 solver
    //  x - input
    //  y - user-managed output vector
    //  dx - timestep
    //  f - user provided lambda function to be evaluated
    template<typename USER_LAMBDA_T>
    UME_FORCE_INLINE float rk4_scalar(
        float x,
        float y,
        float dx,
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
        return y + (1.0f / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
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
            x[i] = randomValue<SCALAR_T>(gen);
            y[i] = randomValue<SCALAR_T>(gen); 
            y_initial[i] = y[i];
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code() {
        int steps = STEP_COUNT; // Number of time steps to calculate
        float timestep = 0.001f;

        auto userFunction = [](auto X, auto Y) { return X * X + Y; };

        for (int i = 0; i < STEP_COUNT; i++) {

            for (int j = 0; j < PROBLEM_LENGTH; j++)
            {
                // Calculate the derivative
                y[j] = rk4_scalar(
                    x[j],
                    y[j],
                    timestep,
                    userFunction);

                // Increment x with the timestep
                x[j] = x[j] + timestep;
            }

            /*
            std::cout << "Iteration: " << i << "\n";
            for (int j = 0; j < 5; j++) {
                std::cout <<
                    "    x(" << x_vec[i] << ") "
                    "y(" << y_vec[i] << ") "
                    "res(" << result_vec[i] << ")\n";
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

        retval += "Scalar (X*X+Y), " +
            ScalarToString<SCALAR_T>::value() + " " +
            std::to_string(PROBLEM_LENGTH);

        return retval;
    }
};

