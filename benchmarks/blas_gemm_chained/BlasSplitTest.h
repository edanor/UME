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
#ifndef BLAS_SPLIT_BENCH_H_
#define BLAS_SPLIT_BENCH_H_

#include <assert.h>

#include <umesimd/UMESimd.h>

#include "MatmulTest.h"
#include "../utilities/UMEScalarToString.h"

#ifdef USE_BLAS

#include "BlasWrapper.h"

template<typename FLOAT_T>
class BlasSplitSingleTest : public MatmulSingleTest<FLOAT_T> {

    void rearrange(FLOAT_T* src, FLOAT_T *dst, FLOAT_T *temp) {
        int N = this->problem_size;
        for (int i = 0; i < N*N; i++) temp[i] = 0;

        // left-upper
        for (int i = 0; i < N / 2; i++) {
            for (int j = 0; j < N / 2; j++) {
                temp[i*N / 2 + j] = src[i*N + j];
            }
        }

        // right upper
        for (int i = 0; i < N/2; i++) {
            for (int j = N / 2; j < N; j++) {
                temp[N*N / 4 + i*N / 2 + (j - N/2)] = src[i*N + j];
            }
        }

        // left lower
        for (int i = N / 2; i < N; i++) {
            for (int j = 0; j < N; j++) {
                temp[N*N / 2 + (i - N/2)*N / 2 + j] = src[i*N + j];
            }
        }

        // right lower
        for (int i = N / 2; i < N; i++) {
            for (int j = N / 2; j < N; j++) {
                temp[3 * N*N / 4 + (i - N / 2)*N / 2 + (j - N/2)] = src[i*N + j];
            }
        }

        for (int i = 0; i < N*N; i++) {
            dst[i] = temp[i];
        }
    }

    void inverse_rearrange(FLOAT_T* src, FLOAT_T* dst, FLOAT_T* temp) {
        int N = this->problem_size;
        // left upper


        // left-upper
        for (int i = 0; i < N / 2; i++) {
            for (int j = 0; j < N / 2; j++) {
                temp[i*N + j] = src[i*N / 2 + j];
            }
        }

        // right upper
        for (int i = 0; i < N / 2; i++) {
            for (int j = N / 2; j < N; j++) {
                temp[i*N + j] = src[N*N / 4 + i*N / 2 + (j - N / 2)];
            }
        }

        // left lower
        for (int i = N / 2; i < N; i++) {
            for (int j = 0; j < N; j++) {
                temp[i*N + j] = src[N*N / 2 + (i - N / 2)*N / 2 + j] ;
            }
        }

        // right lower
        for (int i = N / 2; i < N; i++) {
            for (int j = N / 2; j < N; j++) {
                temp[i*N + j] = src[3 * N*N / 4 + (i - N / 2)*N / 2 + (j - N / 2)];
            }
        }

        for (int i = 0; i < N*N; i++) {
            dst[i] = temp[i];
        }
    }

public:


    BlasSplitSingleTest(int problem_size) : MatmulSingleTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void optional_init() {
        rearrange(this->A, this->A, this->temp0);
        rearrange(this->B, this->B, this->temp0);
    }

    UME_NEVER_INLINE virtual void benchmarked_code() {

        int N = this->problem_size;
        FLOAT_T* a0 = this->A;
        FLOAT_T* a1 = this->A + (N*N / 4);
        FLOAT_T* a2 = this->A + (N*N / 2);
        FLOAT_T* a3 = this->A + (3 * N*N / 4);
        FLOAT_T* b0 = this->B;
        FLOAT_T* b1 = this->B + (N*N / 4);
        FLOAT_T* b2 = this->B + (N*N / 2);
        FLOAT_T* b3 = this->B + (3 * N*N / 4);
        FLOAT_T* r0 = this->R;
        FLOAT_T* r1 = this->R + (N*N / 4);
        FLOAT_T* r2 = this->R + (N*N / 2);
        FLOAT_T* r3 = this->R + (3 * N*N / 4);

        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size / 2, FLOAT_T(1.0f), a0, b0, FLOAT_T(0.0f), r0);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size / 2, FLOAT_T(1.0f), a1, b2, FLOAT_T(1.0f), r0);

        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size / 2, FLOAT_T(1.0f), a0, b1, FLOAT_T(0.0f), r1);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size / 2, FLOAT_T(1.0f), a1, b3, FLOAT_T(1.0f), r1);

        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size / 2, FLOAT_T(1.0f), a2, b0, FLOAT_T(0.0f), r2);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size / 2, FLOAT_T(1.0f), a3, b2, FLOAT_T(1.0f), r2);

        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size / 2, FLOAT_T(1.0f), a2, b1, FLOAT_T(0.0f), r3);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size / 2, FLOAT_T(1.0f), a3, b3, FLOAT_T(1.0f), r3);
    }

    UME_NEVER_INLINE virtual void optional_cleanup() {
        inverse_rearrange(this->A, this->A, this->temp0);
        inverse_rearrange(this->B, this->B, this->temp0);
        inverse_rearrange(this->R, this->R, this->temp0);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "BLAS split single, " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(this->problem_size);
        return retval;
    }
};

template<typename FLOAT_T>
class BlasSplitChainedTest : public MatmulChainedTest<FLOAT_T> {


    void rearrange(FLOAT_T* src, FLOAT_T *dst, FLOAT_T *temp) {
        int N = this->problem_size;
        for (int i = 0; i < N*N; i++) temp[i] = 0;

        // left-upper
        for (int i = 0; i < N / 2; i++) {
            for (int j = 0; j < N / 2; j++) {
                temp[i*N / 2 + j] = src[i*N + j];
            }
        }

        // right upper
        for (int i = 0; i < N/2; i++) {
            for (int j = N / 2; j < N; j++) {
                temp[N*N / 4 + i*N / 2 + (j - N/2)] = src[i*N + j];
            }
        }

        // left lower
        for (int i = N / 2; i < N; i++) {
            for (int j = 0; j < N; j++) {
                temp[N*N / 2 + (i - N/2)*N / 2 + j] = src[i*N + j];
            }
        }

        // right lower
        for (int i = N / 2; i < N; i++) {
            for (int j = N / 2; j < N; j++) {
                temp[3 * N*N / 4 + (i - N / 2)*N / 2 + (j - N/2)] = src[i*N + j];
            }
        }

        for (int i = 0; i < N*N; i++) {
            dst[i] = temp[i];
        }
    }

    void inverse_rearrange(FLOAT_T* src, FLOAT_T* dst, FLOAT_T* temp) {
        int N = this->problem_size;
        // left-upper
        for (int i = 0; i < N / 2; i++) {
            for (int j = 0; j < N / 2; j++) {
                temp[i*N + j] = src[i*N / 2 + j];
            }
        }

        // right upper
        for (int i = 0; i < N / 2; i++) {
            for (int j = N / 2; j < N; j++) {
                temp[i*N + j] = src[N*N / 4 + i*N / 2 + (j - N / 2)];
            }
        }

        // left lower
        for (int i = N / 2; i < N; i++) {
            for (int j = 0; j < N; j++) {
                temp[i*N + j] = src[N*N / 2 + (i - N / 2)*N / 2 + j];
            }
        }

        // right lower
        for (int i = N / 2; i < N; i++) {
            for (int j = N / 2; j < N; j++) {
                temp[i*N + j] = src[3 * N*N / 4 + (i - N / 2)*N / 2 + (j - N / 2)];
            }
        }

        for (int i = 0; i < N*N; i++) {
            dst[i] = temp[i];
        }
    }


public:

    BlasSplitChainedTest(int problem_size) : MatmulChainedTest<FLOAT_T>(problem_size) {
    }
    
    ~BlasSplitChainedTest() {
    }

    UME_NEVER_INLINE virtual void optional_init() {
        // Rearrange A matrix
        //        std::cout << "A:" << std::endl;
        //      printMatrix(this->problem_size, this->A);
        //      std::cout << "A After: " << std::endl;
        rearrange(this->A, this->A, this->temp0);
        //      printMatrix(this->problem_size, this->A);
        // Rearrange B matrix
        //      std::cout << "B:" << std::endl;
        //      printMatrix(this->problem_size, this->B);
        rearrange(this->B, this->B, this->temp0);
        // Rearrange C matrix
        //      std::cout << "C:" << std::endl;
        //      printMatrix(this->problem_size, this->C);
        rearrange(this->C, this->C, this->temp0);
    }


    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        int N = this->problem_size;
        FLOAT_T* a0 = this->A;
        FLOAT_T* a1 = this->A + (N*N/4);
        FLOAT_T* a2 = this->A + (N*N /2);
        FLOAT_T* a3 = this->A + (3* N*N /4);
        FLOAT_T* b0 = this->B;
        FLOAT_T* b1 = this->B + (N*N /4);
        FLOAT_T* b2 = this->B + (N*N /2);
        FLOAT_T* b3 = this->B + (3* N*N /4);
        FLOAT_T* c0 = this->C;
        FLOAT_T* c1 = this->C + (N*N /4);
        FLOAT_T* c2 = this->C + (N*N /2);
        FLOAT_T* c3 = this->C + (3* N*N /4);
        FLOAT_T* t0 = this->temp0;
        FLOAT_T* t1 = this->temp0 + (N*N /4);
        FLOAT_T* t2 = this->temp0 + (N*N /2);
        FLOAT_T* t3 = this->temp0 + (3* N*N /4);

        FLOAT_T* r0 = this->R;
        FLOAT_T* r1 = this->R + (N*N /4);
        FLOAT_T* r2 = this->R + (N*N /2);
        FLOAT_T* r3 = this->R + (3* N*N /4);

//        std::cout << "A0:" << std::endl;
//        printMatrix(this->problem_size/2, a0);

//        std::cout << "B0:" << std::endl;
//        printMatrix(this->problem_size/2, b0);

//        std::cout << "A1:" << std::endl;
//        printMatrix(this->problem_size/2, a1);

//        std::cout << "B2:" << std::endl;
//        printMatrix(this->problem_size/2, b2);

        //GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size, FLOAT_T(1.0f), this->A, this->B, FLOAT_T(0.0f), this->R);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), a0, b0, FLOAT_T(0.0f), t0);

//        std::cout << "t0 1:" << std::endl;
//        printMatrix(this->problem_size/2, t0);

        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), a1, b2, FLOAT_T(1.0f), t0);

//        std::cout << "t0 2:" << std::endl;
//        printMatrix(this->problem_size/2, t0);

//        std::cout << "b1:" << std::endl;
//        printMatrix(this->problem_size/2, b1);

//        std::cout << "b3:" << std::endl;
//        printMatrix(this->problem_size/2, b3);

        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), a0, b1, FLOAT_T(0.0f), t1);
//        std::cout << "t1 1:" << std::endl;
//        printMatrix(this->problem_size/2, t0);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), a1, b3, FLOAT_T(1.0f), t1);
//        std::cout << "t1 2:" << std::endl;
//        printMatrix(this->problem_size/2, t0);

        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), a2, b0, FLOAT_T(0.0f), t2);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), a3, b2, FLOAT_T(1.0f), t2);
        
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), a2, b1, FLOAT_T(0.0f), t3);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), a3, b3, FLOAT_T(1.0f), t3);


//        std::cout << "C0:" << std::endl;
//        printMatrix(this->problem_size/2, c0);

//        std::cout << "C2:" << std::endl;
//        printMatrix(this->problem_size/2, c2);

        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), t0, c0, FLOAT_T(0.0f), r0);
//        std::cout << "r0 1:" << std::endl;
//        printMatrix(this->problem_size/2, r0);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), t1, c2, FLOAT_T(1.0f), r0);
//        std::cout << "r0 2:" << std::endl;
//        printMatrix(this->problem_size/2, r0);

        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), t0, c1, FLOAT_T(0.0f), r1);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), t1, c3, FLOAT_T(1.0f), r1);
        
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), t2, c0, FLOAT_T(0.0f), r2);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), t3, c2, FLOAT_T(1.0f), r2);

        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), t2, c1, FLOAT_T(0.0f), r3);
        GEMM_kernel<FLOAT_T>::blas_gemm(this->problem_size/2, FLOAT_T(1.0f), t3, c3, FLOAT_T(1.0f), r3);

        //std::cout << "Before:" << std::endl;
        //printMatrix(this->problem_size, this->R);
//        std::cout << "R:" << std::endl;
//        printMatrix(this->problem_size, this->R);

    }

    UME_NEVER_INLINE virtual void optional_cleanup() {
        inverse_rearrange(this->A, this->A, this->temp0);
        inverse_rearrange(this->B, this->B, this->temp0);
        inverse_rearrange(this->C, this->C, this->temp0);
        inverse_rearrange(this->R, this->R, this->temp0);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "BLAS split chained, " +
            ScalarToString<FLOAT_T>::value() + " " +
            std::to_string(this->problem_size);
        return retval;
    }
};

#else

// Blas requires external dependencies. This fallback prevents
// compile time error, allowing the user to decide whether to 
// enable BLAS interface or not.
template<typename FLOAT_T>
class BlasSplitSingleTest : public GemvSingleTest<FLOAT_T> {
public:
    BlasSplitSingleTest(int problem_size) : GemvSingleTest<FLOAT_T>(problem_size) {}

    // All the member functions are forced to never inline,
    // so that the compiler doesn't make any opportunistic guesses.
    // Since the cost consuming part of the benchmark, contained
    // in 'benchmarked_code' is measured all at once, the 
    // measurement offset caused by virtual function call should be
    // negligible.
    UME_NEVER_INLINE virtual void initialize() {}
    UME_NEVER_INLINE virtual void benchmarked_code() {}
    UME_NEVER_INLINE virtual void cleanup() {}
    UME_NEVER_INLINE virtual void verify() {}
    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "BLAS split single";
        return retval;
    }
};

template<typename FLOAT_T>
class BlasSplitChainedTest : public GemvChainedTest<FLOAT_T> {
public:
    BlasSplitChainedTest(int problem_size) : GemvChainedTest<FLOAT_T>(problem_size) {}

    // All the member functions are forced to never inline,
    // so that the compiler doesn't make any opportunistic guesses.
    // Since the cost consuming part of the benchmark, contained
    // in 'benchmarked_code' is measured all at once, the 
    // measurement offset caused by virtual function call should be
    // negligible.
    UME_NEVER_INLINE virtual void initialize() {}
    UME_NEVER_INLINE virtual void benchmarked_code() {}
    UME_NEVER_INLINE virtual void cleanup() {}
    UME_NEVER_INLINE virtual void verify() {}
    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "BLAS split chained";
        return retval;
    }
};

#endif

#endif
