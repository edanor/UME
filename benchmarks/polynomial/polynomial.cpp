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

#include <iostream>
#include <memory>

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h> 
#endif

#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <string>

#include <umesimd/UMESimd.h>

#include "ScalarTest.h"
#include "UMESimdTest.h"
#include "UMESimdOmpParallelTest.h"
#include "AVXTest.h"
#include "AVX512Test.h"
#include "OpenmpTest.h"
#include "OpenmpParallelTest.h"

#include "../utilities/TimingStatistics.h"

// Introducing inline assembly forces compiler to generate
#define BREAK_COMPILER_OPTIMIZATION() __asm__ ("NOP");

const int ARRAY_SIZE = 1000000000; // TODO: modify benchmarks to consider peeling effect.

int main(int argc, char **argv)
{
    int MIN_SIZE = 1;
    int MAX_SIZE = 536870912;
    const int ITERATIONS = 5;
    int PROGRESSION = 2;

    BenchmarkHarness harness(argc, argv);

    std::cout << "The result is amount of time it takes to calculate polynomial of\n" 
                 "order 16 (no zero-coefficients) of: " << ARRAY_SIZE << " elements.\n" 
                 "All timing results in nanoseconds. \n"
                 "Speedup calculated with scalar floating point result as reference.\n\n"
                 "SIMD version uses following operations: \n"
                 " ZERO-CONSTR, SET-CONSTR, LOAD, STORE, MULV, FMULADDV, ADDVA\n";

    //for (int i = MIN_SIZE; i <= MAX_SIZE; i*=PROGRESSION) {
        std::string categoryName = std::string("polynomial");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 32));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), ARRAY_SIZE));

        newCategory->registerTest(new ScalarTest<float>(ARRAY_SIZE));
        newCategory->registerTest(new OpenmpTest<float>(ARRAY_SIZE));
        newCategory->registerTest(new OpenmpParallelTest<float>(ARRAY_SIZE));
        newCategory->registerTest(new AVXTest<float>(ARRAY_SIZE));
        newCategory->registerTest(new AVX512Test<float>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<float, 1>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<float, 2>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<float, 4>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<float, 8>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<float, 16>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<float, 32>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<float, 1>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<float, 2>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<float, 4>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<float, 8>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<float, 16>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<float, 32>(ARRAY_SIZE));

        harness.registerTestCategory(newCategory);
    //}

    //for (int i = MIN_SIZE; i <= MAX_SIZE /2; i*= PROGRESSION) {
        categoryName = std::string("polynomial");
        newCategory = new TestCategory(categoryName);
        
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 64));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), ARRAY_SIZE));

        newCategory->registerTest(new ScalarTest<double>(ARRAY_SIZE));
        newCategory->registerTest(new OpenmpTest<double>(ARRAY_SIZE));
        newCategory->registerTest(new OpenmpParallelTest<double>(ARRAY_SIZE));
        newCategory->registerTest(new AVXTest<double>(ARRAY_SIZE));
        newCategory->registerTest(new AVX512Test<double>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<double, 1>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<double, 2>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<double, 4>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<double, 8>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdTest<double, 16>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<double, 1>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<double, 2>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<double, 4>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<double, 8>(ARRAY_SIZE));
        newCategory->registerTest(new UMESimdOmpParallelTest<double, 16>(ARRAY_SIZE));

        harness.registerTestCategory(newCategory);
    //}

    harness.runTests(ITERATIONS);

    return 0;
}
