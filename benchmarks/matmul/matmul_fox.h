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

#ifndef MATMUL_FOX_H_
#define MATMUL_FOX_H_

#include "matmul_common.h"
#include "matmul_naive.h"

// Use Fox's algorithm
template<typename FLOAT_T, int MAT_RANK>
RESULTS<FLOAT_T> test_scalar_fox()
{
    unsigned long long start, end; // Time measurements
    FLOAT_T *A, *B, *C;

    // Allocate alligned to a single scalar
    A = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(MAT_RANK*MAT_RANK*sizeof(FLOAT_T), sizeof(FLOAT_T));
    B = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(MAT_RANK*MAT_RANK*sizeof(FLOAT_T), sizeof(FLOAT_T));
    C = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(MAT_RANK*MAT_RANK*sizeof(FLOAT_T), sizeof(FLOAT_T));

    srand((unsigned int)time(NULL));
    // Initialize arrays with random data
    for (int i = 0; i < MAT_RANK*MAT_RANK; i++)
    {
        // Generate random numbers in range (0.0;1.0)
        A[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        B[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        C[i] = FLOAT_T(0);
    }
    /*
    std::cout << "A: \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << A[i*MAT_RANK + j] << " ";
    }
    std::cout << std::endl;
    }

    std::cout << "\n\nB: \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << B[i*MAT_RANK + j] << " ";
    }
    std::cout << std::endl;
    }
    */
    start = get_timestamp();

    // For each stage
    for (int k = 0; k < MAT_RANK; k++) {
        // For each row in C
        for (int i = 0; i < MAT_RANK; i++) {
            // For each column in C
            for (int j = 0; j < MAT_RANK; j++) {
                int k_mod = (i + k) % MAT_RANK;
                C[i*MAT_RANK + j] += A[i*MAT_RANK + k_mod] * B[k_mod*MAT_RANK + j];
            }
        }
    }

    end = get_timestamp();
    /*
    std::cout << "\n\nC: \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << C[i*MAT_RANK + j] << " ";
    }
    std::cout << std::endl;
    }*/
    FLOAT_T error = calculate_RMS_error_scalar<FLOAT_T, MAT_RANK>(A, B, C);
    //std::cout << "scalar Fox RMS error: " << error << std::endl;

    UME::DynamicMemory::AlignedFree(A);
    UME::DynamicMemory::AlignedFree(B);
    UME::DynamicMemory::AlignedFree(C);

    RESULTS<FLOAT_T> retval;
    retval.elapsed = end - start;
    retval.RMS_error = error;

    return retval;
}

#endif
