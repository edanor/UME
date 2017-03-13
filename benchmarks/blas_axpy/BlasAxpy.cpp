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

#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <string>

#include <umesimd/UMESimd.h>

#include "../utilities/TimingStatistics.h"
#include "../utilities/MeasurementHarness.h"

#include "ScalarTest.h"
#include "BlasTest.h"
#include "UMESimdTest.h"
#include "UMEVectorTest.h"
#include "AsmjitTest.h"
#include "AsmjitUMEVectorTest.h"

int main(int argc, char **argv)
{
    int MIN_SIZE = 1;
    int MAX_SIZE = 268435456;
    int ITERATIONS = 10;
    int PROGRESSION = 2;

    BenchmarkHarness harness(argc, argv);

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
    for (int i = MIN_SIZE; i <= MAX_SIZE; i *= PROGRESSION) {
        std::string categoryName = std::string("BLAS_AXPY_single");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 32));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));

        newCategory->registerTest(new ScalarSingleTest<float>(i));
        // harness.registerTest(new UMEAsmjitSingleTest<float>(i));
        newCategory->registerTest(new UMEVectorAsmjitSingleTest<float>(i));
        newCategory->registerTest(new BlasSingleTest<float>(i));
        newCategory->registerTest(new UMEVectorSingleTest<float>(i));
        newCategory->registerTest(new UMESimdSingleTest<float, 1>(i));
        newCategory->registerTest(new UMESimdSingleTest<float, 2>(i));
        newCategory->registerTest(new UMESimdSingleTest<float, 4>(i));
        newCategory->registerTest(new UMESimdSingleTest<float, 8>(i));
        newCategory->registerTest(new UMESimdSingleTest<float, 16>(i));
        newCategory->registerTest(new UMESimdSingleTest<float, 32>(i));

        harness.registerTestCategory(newCategory);
    }

    // Chained execution (single precision)
    for (int i = MIN_SIZE; i <= MAX_SIZE / 8; i *= PROGRESSION) {
        std::string categoryName = std::string("BLAS_AXPY_chained");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 32));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));

        newCategory->registerTest(new ScalarChainedTest<float>(i));
        //harness.registerTest(new UMEAsmjitChainedTest<float>(i));
        newCategory->registerTest(new UMEVectorAsmjitChainedTest<float>(i));
        newCategory->registerTest(new BlasChainedTest<float>(i));
        newCategory->registerTest(new UMEVectorChainedTest<float>(i));
        newCategory->registerTest(new UMESimdChainedTest<float, 1>(i));
        newCategory->registerTest(new UMESimdChainedTest<float, 2>(i));
        newCategory->registerTest(new UMESimdChainedTest<float, 4>(i));
        newCategory->registerTest(new UMESimdChainedTest<float, 8>(i));
        newCategory->registerTest(new UMESimdChainedTest<float, 16>(i));
        newCategory->registerTest(new UMESimdChainedTest<float, 32>(i));

        harness.registerTestCategory(newCategory);
    }

    // Single execution (double precision)
    for (int i = 1; i <= MAX_SIZE; i *= PROGRESSION) {
        std::string categoryName = std::string("BLAS_AXPY_single");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 64));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));

        newCategory->registerTest(new ScalarSingleTest<double>(i));
        newCategory->registerTest(new BlasSingleTest<double>(i));
        newCategory->registerTest(new UMEVectorSingleTest<double>(i));
        newCategory->registerTest(new UMESimdSingleTest<double, 1>(i));
        newCategory->registerTest(new UMESimdSingleTest<double, 2>(i));
        newCategory->registerTest(new UMESimdSingleTest<double, 4>(i));
        newCategory->registerTest(new UMESimdSingleTest<double, 8>(i));
        newCategory->registerTest(new UMESimdSingleTest<double, 16>(i));

        harness.registerTestCategory(newCategory);
    }

    // Chained execution (double precision)
    for (int i = 1; i <= MAX_SIZE; i *= PROGRESSION) {
        std::string categoryName = std::string("BLAS_AXPY_chained");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 64));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));

        newCategory->registerTest(new ScalarChainedTest<double>(i));
        newCategory->registerTest(new BlasChainedTest<double>(i));
        newCategory->registerTest(new UMEVectorChainedTest<double>(i));
        newCategory->registerTest(new UMESimdChainedTest<double, 1>(i));
        newCategory->registerTest(new UMESimdChainedTest<double, 2>(i));
        newCategory->registerTest(new UMESimdChainedTest<double, 4>(i));
        newCategory->registerTest(new UMESimdChainedTest<double, 8>(i));
        newCategory->registerTest(new UMESimdChainedTest<double, 16>(i));

        harness.registerTestCategory(newCategory);
    }

    harness.runTests(ITERATIONS);
}
