#pragma once


#include <time.h>
#include <string>

#include <umesimd/UMESimd.h>
#include "TimingStatistics.h"

#include <list>

#include "../utilities/ttmath/ttmath/ttmath.h"
#include <tclap/CmdLine.h>

// Each Benchmark can be ran with some parameters, which could make
// comparison with 
class TestParameter {
public:
    virtual std::string getName() = 0;
    virtual std::string getValueAsString() = 0;
};

// This class represents a specific value parameter.
// It should be specialized for non-fundamental types.
template<typename T>
class ValueParameter : public TestParameter {
public:
    T value;
    std::string name;

    ValueParameter(std::string name, T value) :
        value(value),
        name(name)
    {}

    virtual std::string getName() {
        return name;
    }

    virtual std::string getValueAsString() {
        return std::to_string(value);
    }
};

// The test shouldn't allocate large memory buffers before initialize is called.
class Test {
public:
    int iterationOverrider;
    TimingStatistics stats;
    bool validTest;
    ttmath::Big<8, 8> error_norm_bignum;

    Test(bool validTest) : validTest(validTest), iterationOverrider(-1) {}
    Test() : validTest(false), iterationOverrider(-1) {}

    // All the member functions are forced to never inline,
    // so that the compiler doesn't make any opportunistic guesses.
    // Since the cost consuming part of the benchmark, contained
    // in 'benchmarked_code' is measured all at once, the 
    // measurement offset caused by virtual function call should be
    // negligible.
    UME_NEVER_INLINE virtual void initialize() = 0;
    UME_NEVER_INLINE virtual void benchmarked_code() = 0;
    UME_NEVER_INLINE virtual void cleanup() = 0;
    UME_NEVER_INLINE virtual void verify() = 0;
    UME_NEVER_INLINE virtual std::string get_test_identifier() = 0;
};

// Test category represents all tests with directly comparable results.
// All tests in a category should be using the same set of parameters,
// but might differ in actual benchmark implementation.
class TestCategory {
public:
    int iterationOverrider;
    std::string name;
    std::list<Test*> tests;
    std::list<TestParameter*> parameters;
    int iterations;

    TestCategory(std::string name) : name(name), iterationOverrider(-1) {}

    void registerTest(Test *newTest) {
        tests.push_back(newTest);
    }

    void registerParameter(TestParameter* newParam) {
        parameters.push_back(newParam);
    }
};

class BenchmarkHarness {
private:
    // Command line input
    TCLAP::CmdLine *cmd;
    int _argc;
    char **_argv;

    // Command line status:
    //  fastExit flag indicates that no tests should be executed. It
    //  will be set when command line parameters require purely indicative
    //  output (such as '-h' for HELP).
    bool fastExit;

    //  This flag is set when '-i' parameter is passed. It will print a list of all
    //  available tests.
    bool displayTestInfo;
    bool displayCategoriesInfo;
    bool outputJSON;

    bool outputToFile;
    std::string outputFile;
    std::ostream & outputStream;

    // Uncategorized test info
    std::list<Test*> tests;

    // Tests grouped by category
    std::list<TestCategory *> testCategories;

public:

    // Construct without input parameters handling
    BenchmarkHarness() :
        _argc(0),
        _argv(nullptr),
        cmd(nullptr),
        fastExit(false),
        displayTestInfo(false),
        displayCategoriesInfo(false),
        outputJSON(false),
        outputToFile(false),
        outputFile(""),
        outputStream(std::cout)
        {}

    // Construct with input parameters handling
    BenchmarkHarness(int argc, char **argv) :
        _argc(argc),
        _argv(argv),
        fastExit(false),
        displayTestInfo(false),
        displayCategoriesInfo(false),
        outputJSON(false),
        outputToFile(false),
        outputFile(""),
        outputStream(std::cout)
    {
        // Parse the input
        cmd = new TCLAP::CmdLine("Use -h for help.", ' ', "");

        TCLAP::SwitchArg infoFlag("i", "info", "Display list of available tests");
        cmd->add(infoFlag);

        TCLAP::SwitchArg categoriesFlag("c", "categories", "Display list of available test categories");
        cmd->add(categoriesFlag);

        TCLAP::SwitchArg outputJSONFlag("j", "json", "Present output in JSON format");
        cmd->add(outputJSONFlag);

        TCLAP::ValueArg<std::string> fileNameFlag("o", "output", "Set output file name.", false, "","file_name");
        cmd->add(fileNameFlag);

        cmd->parse(_argc, _argv);

        if (infoFlag.getValue()) {
            fastExit = true;
            displayTestInfo = true;
            // Tests are not yet registered.
            // Actual printing has to be delayed.
        }
        else if (categoriesFlag.getValue()) {
            fastExit = true;
            displayCategoriesInfo = true;
            // Tests are not yet registered.
            // Actual printing has to be delayed.
        }

        if (outputJSONFlag.getValue()) {
            outputJSON = true;
        }

        if (fileNameFlag.getValue() != "") {
            outputToFile = true;
            outputFile = fileNameFlag.getValue();
        }
    }

    ~BenchmarkHarness() {
        delete cmd;
    }

    // Register a test without a category
    void registerTest(Test *newTest) {
        tests.push_back(newTest);
    }

    void registerTest(Test *newTest, int iterations) {
        newTest->iterationOverrider = iterations;
        tests.push_back(newTest);
    }

    void registerTestCategory(TestCategory *newCategory) {
        testCategories.push_back(newCategory);
    }

    void registerTestCategory(TestCategory *newCategory, int iterations) {
        newCategory->iterationOverrider = iterations;
        testCategories.push_back(newCategory);
    }


    void runSingleTest(Test* test, int RUNS) {

        unsigned long long start, end;

        int iterations = test->iterationOverrider > 0 ? test->iterationOverrider : RUNS;

        for (int i = 0; i < iterations; i++) {

            // Initialization phase is skipped, as the
            // overhead of memory allocations is not
            // interesting for us
            test->initialize();

            // Start measurement
            start = get_timestamp();
                // The critical fragment of the code being benchmarked
                test->benchmarked_code();

            end = get_timestamp();

            test->verify();
            test->cleanup();

            test->stats.update(end - start);
        }
    }

    void runAllTests(int RUNS) {

        if (outputJSON)
        {
            std::cout << "{ \"test categories\" : [";
        }

        // Execute all categorized tests
        for (auto catIter = testCategories.begin(); catIter != testCategories.end(); catIter++)
        {
            TestCategory* cat = (*catIter);

            if (outputJSON) {
                // Make sure categories are comma separated
                if (catIter != testCategories.begin()) {
                    std::cout << ",";
                }

                std::cout << "\n { \"name\" : \"" << (*catIter)->name << "\",\n";
                std::cout << " \"parameters\" : [";

                for (auto paramIter = (*catIter)->parameters.begin();
                    paramIter != (*catIter)->parameters.end();
                    paramIter++)
                {
                    // Make sure parameters are comma separated
                    if (paramIter != (*catIter)->parameters.begin()) {
                        std::cout << ",";
                    }

                    std::cout << "\n   {"
                        "  \"name\" : \"" << (*paramIter)->getName() << "\"," <<
                        "  \"value\" : \"" << (*paramIter)->getValueAsString() << "\"}";
                }

                std::cout << "\n ]";

                // Add comma between 'parametes' and first 'test' only if tests not empty
                if (cat->tests.size() > 0) {
                    std::cout << ",";
                }
                std::cout << "\n  \"tests\" : [ ";
            }

            for (auto testIter = cat->tests.begin(); testIter != cat->tests.end(); testIter++)
            {
                runSingleTest(*testIter, RUNS);

                if (outputJSON) {
                    // Make sure tests are comma separated.
                    if (testIter != cat->tests.begin())
                    {
                        std::cout << ", ";
                    }

                    std::cout << "\n   { \"name\" : \"" << (*testIter)->get_test_identifier()
                        << "\", \"elapsed\" : \"" << (unsigned long long) (*testIter)->stats.getAverage()
                        << "\", \"stdDev\" : \"" << (unsigned long long) (*testIter)->stats.getStdDev()
                        << "\", \"error\" : \"" << (*testIter)->error_norm_bignum.ToDouble() << "\"}";
                    std::cout << std::flush;
                }
                else {
                    if ((*testIter)->validTest == true) {
                        std::cout << (*testIter)->get_test_identifier()
                            << " Elapsed: " << (unsigned long long) (*testIter)->stats.getAverage()
                            << " (dev: " << (unsigned long long) (*testIter)->stats.getStdDev()
                            << "), error: " << (*testIter)->error_norm_bignum.ToDouble() << ")\n";
                    }
                    else {
                        std::cout << (*testIter)->get_test_identifier()
                            << " RESULTS UNAVAILABLE\n";
                    }
                }

            }

            if (outputJSON)
            {
                std::cout << "  \n  ] \n }";
            }
        }

        if (outputJSON)
        {
            std::cout << "\n ]\n}";
        }

        // Also execute all uncategorized tests
        for (auto testIter = tests.begin(); testIter != tests.end(); testIter++) {
            runSingleTest(*testIter, RUNS);

            if ((*testIter)->validTest == true) {
                std::cout << (*testIter)->get_test_identifier()
                    << " Elapsed: " << (unsigned long long) (*testIter)->stats.getAverage()
                    << " (dev: " << (unsigned long long) (*testIter)->stats.getStdDev()
                    << "), error: " << (*testIter)->error_norm_bignum.ToDouble() << ")\n";
            }
            else {
                std::cout << (*testIter)->get_test_identifier()
                    << " RESULTS UNAVAILABLE\n";
            }
        }
    }

    void runTests(int RUNS) {
        if (displayTestInfo)
        {
            for (auto iter = testCategories.begin(); iter != testCategories.end(); iter++) {
                std::cout << (*iter)->name << " {";
                for (auto paramIter = (*iter)->parameters.begin(); paramIter != (*iter)->parameters.end(); paramIter++) {
                    std::cout << "{" << (*paramIter)->getName() << ": ";
                    std::cout << (*paramIter)->getValueAsString() << "}";
                }
                std::cout << "}\n";

                // Displays all tests
                for (auto test = (*iter)->tests.begin(); test != (*iter)->tests.end(); test++) {
                    std::cout << " " << (*test)->get_test_identifier() << "\n";
                }
            }
        }
        else if (displayCategoriesInfo)
        {
            // Displays all test categories
            for (auto iter = testCategories.begin(); iter != testCategories.end(); iter++) {
                std::cout << (*iter)->name << " {";
                for (auto paramIter = (*iter)->parameters.begin(); paramIter != (*iter)->parameters.end(); paramIter++) {
                    std::cout << "{" << (*paramIter)->getName() << ": ";
                    std::cout << (*paramIter)->getValueAsString() << "}";
                }
                std::cout << "}\n";
            }
        }

        if (fastExit) {
            // Do not run tests when fastExit set.
            return;
        }

        runAllTests(RUNS);
    }

};