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
#include "BlasSplitTest.h"
//#include "UMESimdTest.h"
//#include "UMEVectorTest.h"

int main(int argc, char **argv)
{
    int MIN_SIZE = 512;
    int MAX_SIZE = 4096;
    int PROGRESSION = 2;
    int ITERATIONS = 2;

    BenchmarkHarness harness(argc, argv);

    std::cout <<
        "[Compile with -DUSE_BLAS to enable blas benchmarks (requires BLAS)]\n"
        "\nTODO:\n\n";

    // Single execution (single precision)
    for (int i = MIN_SIZE; i <= MAX_SIZE; i *= PROGRESSION) {
        std::string categoryName = std::string("BLAS_GEMV");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 32));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));

        newCategory->registerTest(new ScalarSingleTest<float>(i));
        newCategory->registerTest(new BlasSingleTest<float>(i));
        newCategory->registerTest(new BlasSplitSingleTest<float>(i));

        harness.registerTestCategory(newCategory);
    }

    // Single execution (double precision)
    for (int i = MIN_SIZE; i <= MAX_SIZE; i *= 2) {
        std::string categoryName = std::string("BLAS_GEMV");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 64));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));

        newCategory->registerTest(new ScalarSingleTest<double>(i));
        newCategory->registerTest(new BlasSingleTest<double>(i));
        newCategory->registerTest(new BlasSplitSingleTest<double>(i));

        harness.registerTestCategory(newCategory);
    }

    // Chained execution (single precision)
    for (int i = MIN_SIZE; i <= MAX_SIZE; i *= PROGRESSION) {
        std::string categoryName = std::string("BLAS_GEMV_chained");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 32));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));

        newCategory->registerTest(new ScalarChainedTest<float>(i));
        newCategory->registerTest(new BlasChainedTest<float>(i));
        newCategory->registerTest(new BlasSplitChainedTest<float>(i));

        harness.registerTestCategory(newCategory);
    }

    // Chained execution (double precision)
    for (int i = MIN_SIZE; i <= MAX_SIZE; i *= 4) {
        std::string categoryName = std::string("BLAS_GEMV_chained");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 64));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));

        newCategory->registerTest(new ScalarChainedTest<double>(i));
        newCategory->registerTest(new BlasChainedTest<double>(i));
        newCategory->registerTest(new BlasSplitChainedTest<double>(i));

        harness.registerTestCategory(newCategory);
    }

    harness.runTests(ITERATIONS);
}
