

// C program to print all permutations with duplicates allowed
#include <fstream>
#include <string>
#include <iostream>
#include <assert.h>

#include <list>

#include "../utilities/TimingStatistics.h"

#include "ProblemGenerator.h"
#include "JsonFormat.h"
#include "DatabaseManager.h"

class Benchmarker {	
private:

    std::string generatePrototypeKernel(ProblemTree * problem) {
        std::list<OP_CLASS_ID> terminals = problem->getTerminalTypes();
        
        std::string code = "#include <umevector/UMEVector.h>\n"
        "\n"
        "#include \"../utilities/MeasurementHarness.h\"\n"
        "#include \"../utilities/UMEScalarToString.h\"\n"
        "\n"
        "#include \"../utilities/ttmath/ttmath/ttmath.h\"\n"
        "\n"
        "template<typename FLOAT_T>\n"
        "class PrototypeTest : public Test {\n"
        "private:\n";
        
        // generate results declarator
        code += "    FLOAT_T *result;\n";
        
        // generate terminals declarators based on the expression/tree
        int currId = 0;
        for(auto iter = terminals.begin(); iter != terminals.end(); iter++) {
            if((*iter) == OP_CLASS_SCALAR) {
                // For scalars generate scalars.
                code += "    FLOAT_T t" + std::to_string(currId) + ";\n";   
            }
            else if ((*iter) == OP_CLASS_VECTOR) {
                // For vectors generate array pointers.
                code += "    FLOAT_T *t" + std::to_string(currId) + ";\n";
            }
            currId++;
        }

        code +=
        "\n"
        "    int problem_size;\n"
        "    static const int OPTIMAL_ALIGNMENT = 64;\n"
        "\n"
        "public:\n"
        "    PrototypeTest(int problem_size) : Test(true), problem_size(problem_size) {}\n"
        "\n"
        "    UME_NEVER_INLINE virtual void initialize() {\n";
        // generate results initializer
        code += "    result=(FLOAT_T*)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);\n";
        // generate terminals initialization
        currId = 0;
        for(auto iter = terminals.begin(); iter != terminals.end(); iter++) {
            if((*iter) == OP_CLASS_SCALAR) {
                code += "        t" + std::to_string(currId) + "=static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);\n";
            }
            else if((*iter) == OP_CLASS_VECTOR) {
                code += "        t" + std::to_string(currId) + "=(FLOAT_T*)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);\n";
            }
            
            currId++;
        }
        
        // generate intialization for vectors
        code += "        for (int i=0; i < problem_size;i++)\n"
                "        {\n";
        currId = 0;
        for(auto iter = terminals.begin(); iter != terminals.end(); iter++) {
            if((*iter) == OP_CLASS_VECTOR) {
                code += "            t" + std::to_string(currId) + "[i]=static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);\n";
            }
            currId++;
        }
        
        code +=
        "        }\n"
        "    }\n"
        "\n"
        "    UME_NEVER_INLINE virtual void benchmarked_code() {\n";
        // generate benchmarking code
        //  generate UME::VECTOR bindings
        
        code += "        UME::VECTOR::Vector<FLOAT_T> result_vec(problem_size, result);\n";
        
        currId = 0;
        for(auto iter = terminals.begin(); iter != terminals.end(); iter++) {
            if((*iter) == OP_CLASS_SCALAR) {
                code += "        UME::VECTOR::Scalar<FLOAT_T> s" + std::to_string(currId) + "(t" + std::to_string(currId) + ");\n";
            }
            else if((*iter) == OP_CLASS_VECTOR) {
                code += "        UME::VECTOR::Vector<FLOAT_T> v" + std::to_string(currId) + "(problem_size, t" + std::to_string(currId) + ");\n";
            }
            currId++;
        }

        code += "        result_vec=" + problem->getExpressionCode() + ";\n";
        
        code +=
        "    }\n"
        "\n"
        "    UME_NEVER_INLINE virtual void cleanup() {\n";
        
        // generate terminals cleanup
        currId = 0;
        for(auto iter = terminals.begin(); iter != terminals.end(); iter++) {
            if((*iter) == OP_CLASS_VECTOR) {
                code += "            UME::DynamicMemory::AlignedFree(t" + std::to_string(currId) + ");\n";
            }
            currId++;
        }
        
        code += 
        "    }\n"
        "\n"
        "    UME_NEVER_INLINE virtual void verify() {\n"
        "        //TODO \n"
        "    }\n"
        "\n"
        "    UME_NEVER_INLINE virtual std::string get_test_identifier() {\n"
        "        return std::string(\"" + problem->getDescriptor() + "\");\n"
        "    }\n"
        "};\n";

        return code;
    }
	
	std::string JSON_FILE_NAME = "benchmark_results.json";
	
public:
    Benchmarker() {}
    ~Benchmarker() {}
    		
	int buildBenchmark(std::string const & exec_file_name)
    {
        std::string command = "time g++ test_kernel.cpp -std=c++11 -O3 -mavx2 -o ";
        command += exec_file_name;
     
        std::cout << "Execute: " << command << std::endl;
        int retval = system(command.c_str());
        std::cout << "Returned: " << retval << "\n";
		
		return retval;
    }    
	
    int executeBenchmark(std::string const & exec_file_name)
    {
        std::string command = "time taskset -c 3 ./";
        command += exec_file_name;
		command += " -o " + JSON_FILE_NAME + " -j ";
        std::cout << command << std::endl;
        int retval = system(command.c_str());
        std::cout << "Returned: " << retval << "\n";
		
		return retval;
    }
    
    int parseBenchmarkResults(std::string const & results_file_name) {
        // TODO: benchmark results should've been written to the .json results file
		JsonFormat inJson(results_file_name);
		
		DatabaseManager db("measurements.db");
		
		int retval = db.insert(inJson.testResults);
		return retval;
    }
    
    void cleanBenchmark(std::string const & exec_file_name)
    {
        std::string command = "rm " + exec_file_name;
        std::cout << "Execute: " << command << std::endl;
		int retval = system(command.c_str());
		
		command = "rm benchmark_results.json";
		std::cout << "Execute: " << command << std::endl;
		retval = system(command.c_str());
    }
    
    int runBenchmark(ProblemTree * problem) {
        std::string kernelCode = generatePrototypeKernel(problem);
        std::ofstream kernelFile("PrototypeTest.h");
        kernelFile << kernelCode;
        kernelFile.close();        
        
        std::string benchmarkFileName = "benchmark.out";
        
        int retval = buildBenchmark(benchmarkFileName);
		if(retval != 0) {
			std::cout << "Failed to build benchmark.\n";
			goto exit;
		}

        retval = executeBenchmark(benchmarkFileName);
		if(retval != 0) {
			std::cout << "Failed to execute benchmark.\n";
			goto exit;
		}
		
		retval = parseBenchmarkResults("benchmark_results.json");
		if(retval != 0) {
			std::cout << "Failed to parse benchmark.\n";
		}
		
	exit:
        cleanBenchmark(benchmarkFileName);
		
		return retval;
    }
};

int main()
{
    //Benchmarker bench;
    
    ProblemGenerator gen;
	int executions = 1000;
    
    for(int i = 0; i < executions; i++) {
        ProblemTree * problem = gen.getRandomProblem();
        std::cout << problem->getDescriptor().c_str() << std::endl;
        problem->print();        
        std::cout << "\n";
        std::cout << "-------\n";

        Benchmarker bench;
        bench.runBenchmark(problem);
    }

    return 0;
}
