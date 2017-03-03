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

#include "AxpyTest.h"

template<typename FLOAT_T>
class UMEVectorSingleTest : public AxpySingleTest<FLOAT_T>{
public:
    UMEVectorSingleTest(int problem_size) : AxpySingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        UME::VECTOR::Vector<FLOAT_T> x_vec(this->problem_size, this->x);
        UME::VECTOR::Vector<FLOAT_T> y_vec(this->problem_size, this->y);
        UME::VECTOR::Scalar<FLOAT_T> alpha_s(this->alpha);

        y_vec = alpha_s*x_vec + y_vec;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "UME::VECTOR single, (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(this->problem_size);
        return retval;
    }
};

template<typename FLOAT_T>
class UMEVectorChainedTest : public AxpyChainedTest<FLOAT_T> {
public:
    UMEVectorChainedTest(int problem_size) : AxpyChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        UME::VECTOR::Vector<FLOAT_T> x0_vec(this->problem_size, this->x0);
        UME::VECTOR::Vector<FLOAT_T> x1_vec(this->problem_size, this->x1);
        UME::VECTOR::Vector<FLOAT_T> x2_vec(this->problem_size, this->x2);
        UME::VECTOR::Vector<FLOAT_T> x3_vec(this->problem_size, this->x3);
        UME::VECTOR::Vector<FLOAT_T> x4_vec(this->problem_size, this->x4);
        UME::VECTOR::Vector<FLOAT_T> x5_vec(this->problem_size, this->x5);
        UME::VECTOR::Vector<FLOAT_T> x6_vec(this->problem_size, this->x6);
        UME::VECTOR::Vector<FLOAT_T> x7_vec(this->problem_size, this->x7);
        UME::VECTOR::Vector<FLOAT_T> x8_vec(this->problem_size, this->x8);
        UME::VECTOR::Vector<FLOAT_T> x9_vec(this->problem_size, this->x9);
        UME::VECTOR::Vector<FLOAT_T> y_vec(this->problem_size, this->y);

        UME::VECTOR::Scalar<FLOAT_T> alpha0(this->alpha[0]);
        UME::VECTOR::Scalar<FLOAT_T> alpha1(this->alpha[1]);
        UME::VECTOR::Scalar<FLOAT_T> alpha2(this->alpha[2]);
        UME::VECTOR::Scalar<FLOAT_T> alpha3(this->alpha[3]);
        UME::VECTOR::Scalar<FLOAT_T> alpha4(this->alpha[4]);
        UME::VECTOR::Scalar<FLOAT_T> alpha5(this->alpha[5]);
        UME::VECTOR::Scalar<FLOAT_T> alpha6(this->alpha[6]);
        UME::VECTOR::Scalar<FLOAT_T> alpha7(this->alpha[7]);
        UME::VECTOR::Scalar<FLOAT_T> alpha8(this->alpha[8]);
        UME::VECTOR::Scalar<FLOAT_T> alpha9(this->alpha[9]);

        y_vec = y_vec + alpha0*x0_vec + alpha1*x1_vec +
            alpha2*x2_vec + alpha3*x3_vec +
            alpha4*x4_vec + alpha5*x5_vec +
            alpha6*x6_vec + alpha7*x7_vec +
            alpha8*x8_vec + alpha9*x9_vec;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "UME::VECTOR chained, (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(this->problem_size);
        return retval;
    }
};
