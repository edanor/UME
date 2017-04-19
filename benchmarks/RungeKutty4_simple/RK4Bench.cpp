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

#include <umevector/UMEVector.h>

#include <random>

#include "ScalarTest.h"
#include "UMESimdTest.h"
#include "UMEVectorTest.h"

int main(int argc, char** argv)
{
    BenchmarkHarness harness(argc, argv);
    int MIN_SIZE = 1;
    int MAX_SIZE = 10000000;
    int STEP_COUNT = 1000;
    int PROGRESSION = 10;

    for (int i = MIN_SIZE; i <= MAX_SIZE; i *= PROGRESSION)
    {
        std::string categoryName = std::string("RK4_bench1");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 64));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));
        newCategory->registerParameter(new ValueParameter<int>(std::string("step_count"), STEP_COUNT));

        newCategory->registerTest(new ScalarTest<double>(i, STEP_COUNT));
        newCategory->registerTest(new UMESimdTest<double, DefaultStride<double>::value>(i, STEP_COUNT));
        newCategory->registerTest(new UMEVectorTest<double>(i, STEP_COUNT));

        harness.registerTestCategory(newCategory);
    }

    for (int i = MIN_SIZE; i <= MAX_SIZE; i *= PROGRESSION)
    {
        std::string categoryName = std::string("RK4_bench1");
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 32));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));
        newCategory->registerParameter(new ValueParameter<int>(std::string("step_count"), STEP_COUNT));

        newCategory->registerTest(new ScalarTest<float>(i, STEP_COUNT));
        newCategory->registerTest(new UMESimdTest<float, DefaultStride<float>::value>(i, STEP_COUNT));
        newCategory->registerTest(new UMEVectorTest<float>(i, STEP_COUNT));

        harness.registerTestCategory(newCategory);
    }
    harness.runAllTests(10);

    return 0;
}