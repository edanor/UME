#include <iostream>
#include <fstream>

#include "../utilities/TimingStatistics.h"
#include "../utilities/MeasurementHarness.h"

#include "PrototypeTest.h"

int main(int argc, char **argv)
{
    int MIN_SIZE = 1;
    int MAX_SIZE = 16*1024*1024;
    int PROGRESSION = 2;
    int ITERATIONS = 20;

    BenchmarkHarness harness(argc, argv);

    // Chained execution (single precision)
    for (int i = MIN_SIZE; i <= MAX_SIZE; i *= PROGRESSION) {
		
        std::string categoryName = "UME::VECTOR";
        TestCategory *newCategory = new TestCategory(categoryName);
        newCategory->registerParameter(new ValueParameter<int>(std::string("precision"), 32));
        newCategory->registerParameter(new ValueParameter<int>(std::string("problem_size"), i));
        newCategory->registerParameter(new ValueParameter<int>(std::string("executions"), ITERATIONS));

        Test* test = new PrototypeTest<float>(i);
        newCategory->registerTest(test);

        harness.registerTestCategory(newCategory);
    }
    harness.runTests(ITERATIONS);
}
