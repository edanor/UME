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

#include <umevector/UMEVector.h>

#include "GemvTest.h"

template<typename FLOAT_T>
class UMEVectorSingleTest : public GemvSingleTest<FLOAT_T> {
public:
    UMEVectorSingleTest(int problem_size) : GemvSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE void evaluateRow(int N, FLOAT_T alpha, FLOAT_T* A_row, FLOAT_T* x, FLOAT_T beta, FLOAT_T* y)
    {
        UME::VECTOR::Vector<FLOAT_T> A_vec(N, A_row);
        UME::VECTOR::Vector<FLOAT_T> x_vec(N, x);

        auto t0 = (A_vec * x_vec).hadd(); // dot product
        FLOAT_T t1 = beta * (*y);
        auto t2 = alpha * t0 + t1;
        UME::VECTOR::MonadicEvaluator eval(y, t2); // evaluate the reduction operation
    }

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        // Traverse rows of A
        /*for (int i = 0; i < this->problem_size; i++)
        {
            int row_offset = i * this->problem_size;
            UME::VECTOR::Vector<FLOAT_T> A_vec(this->problem_size, &this->A[row_offset]);
            UME::VECTOR::Vector<FLOAT_T> x_vec(this->problem_size, this->x);

            auto t0 = (A_vec * x_vec).hadd(); // dot product
            FLOAT_T t1 = this->beta * this->y[i];
            auto t2 = this->alpha * t0 + t1;
            UME::VECTOR::MonadicEvaluator eval((FLOAT_T*)&this->y[i], t2); // evaluate the reduction operation
        }*/
        for (int i = 0; i < this->problem_size; i++)
        {
            int row_offset = i * this->problem_size;
            evaluateRow(i, this->alpha, &this->A[row_offset], this->x, this->beta, &this->y[i]);
        }
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "UME::VECTOR single";
        return retval;
    }
};

#include "umevector/utilities/ExpressionPrinter.h"

template<typename FLOAT_T>
void printArray(int N, FLOAT_T* arr, std::string const & name) {
    std::cout << name << "\n ";
    for (int i = 0; i < N; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";
}

template<typename FLOAT_T>
void printMatrix(int N, int M, FLOAT_T* arr, std::string const & name) {
    std::cout << name << "\n";
    for (int i = 0; i < M; i++) {
        printArray(N, &arr[i*N], std::string(""));
    }
    std::cout << "\n";
}

template<typename FLOAT_T>
class UMEVectorChainedTest : public GemvChainedTest<FLOAT_T> {
public:
    UMEVectorChainedTest(int problem_size) : GemvChainedTest<FLOAT_T>(problem_size) {}

    // WA: problem with Inlining
    UME_NEVER_INLINE void evaluateForRow(
        int N,
        FLOAT_T* A0_row,
        FLOAT_T* A1_row,
        FLOAT_T* A2_row,
        FLOAT_T* A3_row,
        FLOAT_T* A4_row,
        FLOAT_T* alphas,
        FLOAT_T* x0,
        FLOAT_T* x1,
        FLOAT_T* x2,
        FLOAT_T* x3,
        FLOAT_T* x4,
        FLOAT_T* betas,
        FLOAT_T* y)
    {
        // Load rows of A's.
        UME::VECTOR::Vector<FLOAT_T> A0_vec(N, A0_row);
        UME::VECTOR::Vector<FLOAT_T> A1_vec(N, A1_row);
        UME::VECTOR::Vector<FLOAT_T> A2_vec(N, A2_row);
        UME::VECTOR::Vector<FLOAT_T> A3_vec(N, A3_row);
        UME::VECTOR::Vector<FLOAT_T> A4_vec(N, A4_row);

        // Load x vectors.
        UME::VECTOR::Vector<FLOAT_T> x0_vec(N, x0);
        UME::VECTOR::Vector<FLOAT_T> x1_vec(N, x1);
        UME::VECTOR::Vector<FLOAT_T> x2_vec(N, x2);
        UME::VECTOR::Vector<FLOAT_T> x3_vec(N, x3);
        UME::VECTOR::Vector<FLOAT_T> x4_vec(N, x4);

        /*x0_vec.elements = this->x0;
        x1_vec.elements = this->x1;
        x2_vec.elements = this->x2;
        x3_vec.elements = this->x3;
        x4_vec.elements = this->x4;*/

        // This code touches only one row of each array per iteration!
        // y vector is accessed therefore exactly once.
        auto y_i_0 = (alphas[0] * ((A0_vec * x0_vec).hadd())) + (this->beta[0] * (*y));
        auto y_i_1 = (alphas[1] * ((A1_vec * x1_vec).hadd())) + (this->beta[1] * y_i_0);
        auto y_i_2 = (alphas[2] * ((A2_vec * x2_vec).hadd())) + (this->beta[2] * y_i_1);
        auto y_i_3 = (alphas[3] * ((A3_vec * x3_vec).hadd())) + (this->beta[3] * y_i_2);
        auto y_i_4 = (alphas[4] * ((A4_vec * x4_vec).hadd())) + (this->beta[4] * y_i_3);

        //ExpressionPrinter<decltype(y_i_4)> printer(y_i_4);
        UME::VECTOR::MonadicEvaluator eval(y, y_i_4);
    }

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        /*printArray(this->problem_size, this->x0, std::string("x0"));
        printArray(this->problem_size, this->x1, std::string("x1"));
        printArray(this->problem_size, this->x2, std::string("x2"));
        printArray(this->problem_size, this->x3, std::string("x3"));
        printArray(this->problem_size, this->x4, std::string("x4"));

        printMatrix(this->problem_size, this->problem_size, this->A0, "A0");
        printMatrix(this->problem_size, this->problem_size, this->A1, "A1");
        printMatrix(this->problem_size, this->problem_size, this->A2, "A2");
        printMatrix(this->problem_size, this->problem_size, this->A3, "A3");
        printMatrix(this->problem_size, this->problem_size, this->A4, "A4");
        */
        // Traverse rows of A's
        /*for (int i = 0; i < this->problem_size; i++)
        {
            int row_offset = i * this->problem_size;
            // Load rows of A's.
            UME::VECTOR::Vector<FLOAT_T, UME_DYNAMIC_LENGTH, 1> A0_vec(this->problem_size, &this->A0[row_offset]);
            UME::VECTOR::Vector<FLOAT_T, UME_DYNAMIC_LENGTH, 1> A1_vec(this->problem_size, &this->A1[row_offset]);
            UME::VECTOR::Vector<FLOAT_T, UME_DYNAMIC_LENGTH, 1> A2_vec(this->problem_size, &this->A2[row_offset]);
            UME::VECTOR::Vector<FLOAT_T, UME_DYNAMIC_LENGTH, 1> A3_vec(this->problem_size, &this->A3[row_offset]);
            UME::VECTOR::Vector<FLOAT_T, UME_DYNAMIC_LENGTH, 1> A4_vec(this->problem_size, &this->A4[row_offset]);

            // Load x vectors.
            UME::VECTOR::Vector<FLOAT_T, UME_DYNAMIC_LENGTH, 1> x0_vec(this->problem_size, this->x0);
            UME::VECTOR::Vector<FLOAT_T, UME_DYNAMIC_LENGTH, 1> x1_vec(this->problem_size, this->x1);
            UME::VECTOR::Vector<FLOAT_T, UME_DYNAMIC_LENGTH, 1> x2_vec(this->problem_size, this->x2);
            UME::VECTOR::Vector<FLOAT_T, UME_DYNAMIC_LENGTH, 1> x3_vec(this->problem_size, this->x3);
            UME::VECTOR::Vector<FLOAT_T, UME_DYNAMIC_LENGTH, 1> x4_vec(this->problem_size, this->x4);

            // This code touches only one row of each array per iteration!
            // y vector is accessed therefore exactly once.
            auto y_i_0 = (this->alpha[0] * ((A0_vec * x0_vec).hadd())) + (this->beta[0] * this->y[i]);
            auto y_i_1 = (this->alpha[1] * ((A1_vec * x1_vec).hadd())) + (this->beta[1] * y_i_0);
            auto y_i_2 = (this->alpha[2] * ((A2_vec * x2_vec).hadd())) + (this->beta[2] * y_i_1);
            auto y_i_3 = (this->alpha[3] * ((A3_vec * x3_vec).hadd())) + (this->beta[3] * y_i_2);
            auto y_i_4 = (this->alpha[4] * ((A4_vec * x4_vec).hadd())) + (this->beta[4] * y_i_3);

            //ExpressionPrinter<decltype(y_i_4)> printer(y_i_4);
            UME::VECTOR::MonadicEvaluator eval((FLOAT_T*)&this->y[i], y_i_4);
        }*/

        for (int i = 0; i < this->problem_size; i++)
        {
            int rowOffset = i*this->problem_size;
            evaluateForRow(
                this->problem_size,
                &this->A0[rowOffset],
                &this->A1[rowOffset],
                &this->A2[rowOffset],
                &this->A3[rowOffset],
                &this->A4[rowOffset],
                this->alpha,
                this->x0,
                this->x1,
                this->x2,
                this->x3,
                this->x4,
                this->beta,
                &this->y[i]);
        }
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "UME::VECTOR chained";
        return retval;
    }
};
