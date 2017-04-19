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

#include "RK4Test.h"

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMERandomValues.h"
#include "../utilities/UMEScalarToString.h"

template<typename SCALAR_T>
class ScalarTest : public RK4Test<SCALAR_T> {
private:
    static const int OPTIMAL_ALIGNMENT = 64;
    // Scalar RK4 solver
    //  x - input
    //  y - user-managed output vector
    //  dx - timestep
    //  f - user provided lambda function to be evaluated
    template<typename USER_LAMBDA_T>
    UME_FORCE_INLINE SCALAR_T rk4_scalar(
        SCALAR_T x,
        SCALAR_T y,
        SCALAR_T dx,
        USER_LAMBDA_T & f)
    {
        SCALAR_T halfdx = dx * 0.5f;

        // Implement RK4 algorithm - very straightforward process.
        // the user function is here attached as a fragment of computation
        // graph, and it can be optimized for each 'k' independantly.
        SCALAR_T k1 = dx * f(x, y);
        SCALAR_T k2 = dx * f(x + halfdx, y + k1 * halfdx);
        SCALAR_T k3 = dx * f(x + halfdx, y + k2 * halfdx);
        SCALAR_T k4 = dx * f(x + dx, y + k3 * dx);

        // Merge into full computational graph and start evaluation.
        return y + (1.0f / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    }

public:

    ScalarTest(int problem_size, int step_count) : RK4Test<SCALAR_T>(problem_size, step_count) {}

    UME_NEVER_INLINE virtual void benchmarked_code() {
        float timestep = 0.001f;

        //auto userFunction = [](auto X, auto Y) { return std::exp(X)*std::sin(Y); };
        auto userFunction = [](auto X, auto Y) { return X*X + Y; };

        for (int i = 0; i < this->step_count; i++) {

            for (int j = 0; j < this->problem_size; j++)
            {
                // Calculate the derivative
                this->y[j] = rk4_scalar(
                    this->x[j],
                    this->y[j],
                    timestep,
                    userFunction);

                // Increment x with the timestep
                this->x[j] = this->x[j] + timestep;
            }
        }
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "Scalar (X*X+Y)";
        return retval;
    }
};

