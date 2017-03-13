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

#include <umevector/UMEVector.h>
#include "../utilities/MeasurementHarness.h"

template<typename FLOAT_T>
class UMEVectorSingleTest : public DotSingleTest<FLOAT_T> {
public:
    UMEVectorSingleTest(int problem_size) : DotSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        UME::VECTOR::Vector<FLOAT_T> x_vec(this->problem_size, this->x);
        UME::VECTOR::Vector<FLOAT_T> y_vec(this->problem_size, this->y);

        this->dot_result = (x_vec * y_vec).hadd();
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "UME::VECTOR single";
        return retval;
    }
};

template<typename FLOAT_T>
class UMEVectorChainedTest : public DotChainedTest<FLOAT_T> {
public:
    UMEVectorChainedTest(int problem_size) : DotChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        UME::VECTOR::Vector<FLOAT_T> x0_vec(this->problem_size, this->x0);
        UME::VECTOR::Vector<FLOAT_T> x1_vec(this->problem_size, this->x1);
        UME::VECTOR::Vector<FLOAT_T> y0_vec(this->problem_size, this->y0);
        UME::VECTOR::Vector<FLOAT_T> y1_vec(this->problem_size, this->y1);

        auto t0 = y0_vec.adda(this->alpha0*x0_vec); // equivalent of Y = aX + Y (AXPY)
        auto t1 = y1_vec.adda(this->alpha1*x1_vec);
        this->dot_result = (t0 * t1).hadd();
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "UME::VECTOR dot(axpy(x0, y0), axpy(x1, y1))";
        return retval;
    }
};