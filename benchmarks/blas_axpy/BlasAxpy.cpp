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

#include <iostream>
#include <memory>

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h> 
#endif

#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <string>

#include <umesimd/UMESimd.h>

#include "../utilities/TimingStatistics.h"
#include "../utilities/MeasurementHarness.h"

// Introducing inline assembly forces compiler to generate
#define BREAK_COMPILER_OPTIMIZATION() __asm__ ("NOP");

#include "ScalarTest.h"
#include "BlasTest.h"
#include "UMEVectorTest.h"

int main()
{
    int MAX_SIZE = 100000000;
    int ITERATIONS = 10;

    BenchmarkHarness harness;

    std::cout <<
        "[Compile with -DUSE_BLAS to enable blas benchmarks (requires BLAS)]\n"
        "\n"
        "This benchmark measures execution of Y = a * X + Y (AXPY) kernel.\n"
        "Two modes are being measured:\n"
        " - single kernel execution (as defined by BLAS)\n"
        " - chained execution, where 10 AXPY kernels are used in a daisy-chain\n\n"
        "Usually BLAS primitives' performance is shown with a single highly tuned\n"
        "implementation in mind. However what is ommited is the ability to exploit\n"
        "data locality in context of more than a single BLAS call.\n"
        "UME::VECTOR version shows that treatment of vector programs using\n"
        "expressions instead of kernels can give additional performance boost.\n\n";

    // Single execution (single precision)
    for (int i = 1; i <= MAX_SIZE; i *= 100) {
        harness.registerTest(new ScalarSingleTest<float>(i));
        harness.registerTest(new BlasSingleTest<float>(i));
        harness.registerTest(new UMEVectorSingleTest<float>(i));
    }

    // Single execution (double precision)
    for (int i = 1; i <= MAX_SIZE; i *= 100) {
        harness.registerTest(new ScalarSingleTest<double>(i));
        harness.registerTest(new BlasSingleTest<double>(i));
        harness.registerTest(new UMEVectorSingleTest<double>(i));
    }


    // Chained execution (single precision)
    for (int i = 1; i <= MAX_SIZE; i *= 100) {
        harness.registerTest(new ScalarChainedTest<double>(i));
        harness.registerTest(new BlasChainedTest<double>(i));
        harness.registerTest(new UMEVectorSingleTest<double>(i));
    }

    // Chained execution (double precision)
    for (int i = 1; i <= MAX_SIZE; i *= 100) {
        harness.registerTest(new ScalarChainedTest<double>(i));
        harness.registerTest(new BlasChainedTest<double>(i));
        harness.registerTest(new UMEVectorSingleTest<double>(i));
    }

    harness.runAllTests(ITERATIONS);
}
