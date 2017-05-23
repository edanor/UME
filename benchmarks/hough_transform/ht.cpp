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
//#include <umesimd/utilities/ignore_warnings_inline_noinline.h>

#include "ScalarHTTest.h"
#include "UmesimdHTTest.h"
#include "UmevectorHTTest.h"
#include "AVXHTTest.h"
#include "AVX512HTTest.h"

int main(int argc, char **argv)
{
   /* uint32_t test[15] = { 1, 2, 3, 6, 8, 9, 3, 2, 1, 1, 12, 8, 7, 21, 20 };

    Vector<uint32_t> A(15, test);

    auto t0 = A.hmax();
    uint32_t res = 0;
    MonadicEvaluator eval(&res, t0);*/


    const int ITERATIONS = 10;

    srand ((unsigned int)time(NULL));

    std::vector<std::string> inputFileNames = {
        //"data/inputs/lines0.bmp",
        "data/inputs/lines1.bmp",
        //"data/inputs/lines9.bmp"
        //"data/inputs/lines12.bmp"
        //"data/inputs/lines1_noise.bmp"
    };

    std::vector<std::string> resultFileNames = {
        //"data/gold/lines0_result.bmp",
        "data/gold/lines1_result.bmp",
        //"data/gold/lines9_result.bmp"
        //"data/gold/lines12_result.bmp"
    };

    std::vector<std::string> categoryName = {
        //"HT lines 0",
        "HT lines 1",
        "HT lines 9"
    };

    BenchmarkHarness harness(argc, argv);

    std::cout <<
        "This benchmark uses the Hough Transform algorithm to find lines in bitmap images.\n\n"
        "All timing results in nanoseconds. \n"
        "Speedup calculated with scalar floating point result as reference.\n\n";

    // Each test input creates a new test category

    for (uint32_t i = 0; i < inputFileNames.size(); i++) {
        TestCategory *newCategory = new TestCategory(categoryName[i]);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 32));
        //newCategory->registerTest(new ScalarHTTest<float>(inputFileNames[i], resultFileNames[i]));
        newCategory->registerTest(new UmevectorHTTest<float>(inputFileNames[i], resultFileNames[i]));
        newCategory->registerTest(new UmesimdHTTest<float, 8>(inputFileNames[i], resultFileNames[i]));
        newCategory->registerTest(new UmesimdHTTest<float, 16>(inputFileNames[i], resultFileNames[i]));
        newCategory->registerTest(new UmesimdHTTest<float, 32>(inputFileNames[i], resultFileNames[i]));
        newCategory->registerTest(new AvxHTTest<float>(inputFileNames[i], resultFileNames[i]));
        //newCategory->registerTest(new Avx512HTTest<float>(inputFileNames[i], resultFileNames[i]));
        newCategory->registerTest(new ScalarHTTest<float>(inputFileNames[i], resultFileNames[i]));
        harness.registerTestCategory(newCategory);
    }

    harness.runTests(ITERATIONS);
    return 0;
}

#include <umesimd/utilities/ignore_warnings_pop.h>
