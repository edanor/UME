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

#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <string>

#include <umesimd/utilities/ignore_warnings_push.h>
#include <umesimd/utilities/ignore_warnings_unused_but_set.h>
#include <umesimd/utilities/ignore_warnings_inline_noinline.h>

#include "ScalarAverageTest.h"
#include "AVXIntrinsicsAverageTest.h"
#include "AVX512IntrinsicsAverageTest.h"
#include "UmesimdAverageTest.h"
#include "UmevectorAverageTest.h"

int main(int argc, char **argv)
{
    const int ITERATIONS = 10;
    const int MIN_PROBLEM_SIZE = 1;
    const int MAX_PROBLEM_SIZE = 1073741824;
    const int PROGRESSION = 2;
    const int PROBLEM_SIZE_OFFSET = 7; // We will use this value to offset the problem size a little.
                                       // The point is to force at least some of the runs to take 
                                       // remainder calculation into consideration.

    srand ((unsigned int)time(NULL));

    BenchmarkHarness harness(argc, argv);

    std::cout <<
        "All timing results in nanoseconds. \n"
        "Speedup calculated with scalar floating point result as reference.\n\n"
        "SIMD version uses following operations: \n"
        " ZERO-CONSTR, SET-CONSTR, LOAD, ADDA, STORE\n";

    for(int i = MIN_PROBLEM_SIZE; i < MAX_PROBLEM_SIZE; i*=PROGRESSION) {
        std::string categoryName = std::string("average");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 32));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i+PROBLEM_SIZE_OFFSET));

        newCategory->registerTest(new ScalarAverageTest<float>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new AVXAverageTest<float>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new AVX512AverageTest<float>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<float, 1>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<float, 2>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<float, 4>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<float, 8>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<float, 16>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<float, 32>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmevectorAverageTest<float>(i+PROBLEM_SIZE_OFFSET));

        harness.registerTestCategory(newCategory);
    }
    
    for(int i = MIN_PROBLEM_SIZE; i < MAX_PROBLEM_SIZE/PROGRESSION; i*=PROGRESSION) {
        std::string categoryName = std::string("average");
        TestCategory *newCategory = new TestCategory(categoryName);newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 64));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i+PROBLEM_SIZE_OFFSET));

        newCategory->registerTest(new ScalarAverageTest<double>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new AVXAverageTest<double>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new AVX512AverageTest<double>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<double, 1>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<double, 2>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<double, 4>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<double, 8>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmesimdAverageTest<double, 16>(i+PROBLEM_SIZE_OFFSET));
        newCategory->registerTest(new UmevectorAverageTest<double>(i+PROBLEM_SIZE_OFFSET));

        harness.registerTestCategory(newCategory);
    }

    harness.runTests(ITERATIONS);

    return 0;
}

#include <umesimd/utilities/ignore_warnings_pop.h>
