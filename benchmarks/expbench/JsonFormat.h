#pragma once

#include <string>
#include <fstream>
#include "../utilities/json/src/json.hpp"

class ResultDesc {
public:
	std::string name;
	uint64_t elapsed;
	uint64_t stdDev;
	//double error;
	
	ResultDesc(std::string const & name, uint64_t elapsed, uint64_t stdDev)
		: name(name), elapsed(elapsed), stdDev(stdDev)
	{}
};

class ParameterDesc {
public:
	std::string name;
	int64_t value;
	
	ParameterDesc(std::string const & name, uint64_t value) 
		: name(name), value(value)
	{}
};

class TestDesc {
public:
	std::string name;
	std::list<ParameterDesc*> parameters;
	std::list<ResultDesc*> results;
	
	TestDesc(std::string const & name) : name(name) {}
	~TestDesc() {
		parameters.clear();
		results.clear();
	}
	
	void pushParameters(ParameterDesc* params) {
		parameters.push_back(params);
	}
	
	void pushResults(ResultDesc* result) {
		this->results.push_back(result);
	}
	
	void print() {
		std::cout << name << "\n";
		for(auto iter = parameters.begin(); iter != parameters.end(); iter++) {
			std::cout << "    Parameter: " << (*iter)->name  << " " << (*iter)->value << std::endl;
		}
		for(auto iter = results.begin(); iter != results.end(); iter++) {
			std::cout << "    Result: " << (*iter)->name  << " " << (*iter)->elapsed << " " << (*iter)->stdDev << std::endl;
		}
	}
};

class JsonFormat {
public:
	std::list<TestDesc*> testResults;
	
	JsonFormat(std::string const & fileName) {
		std::ifstream in(fileName.c_str());
		nlohmann::json jsonObj(in);
		
		for(auto categoriesIter = jsonObj.begin(); categoriesIter != jsonObj.end(); categoriesIter++) {
			for(auto testIter = (*categoriesIter).begin(); testIter != (*categoriesIter).end(); testIter++) {
				std::string testName = (*testIter)["name"];
				TestDesc *test = new TestDesc(testName);
				
				for(auto paramIter = (*testIter)["parameters"].begin(); paramIter != (*testIter)["parameters"].end(); paramIter++) {
					std::cout << "Param: " << *paramIter << std::endl;
					
					ParameterDesc *newParam = 
						new ParameterDesc(
							(*paramIter)["name"],
							std::strtoul((*paramIter)["value"].get<std::string>().c_str(), NULL, 0)); 
					
					test->pushParameters(newParam);
				}
				
				for(auto resultIter = (*testIter)["tests"].begin(); resultIter != (*testIter)["tests"].end(); resultIter++) {
					std::cout << "Result: " << *resultIter << std::endl;
					
					ResultDesc *newResult = 
						new ResultDesc(
							(*resultIter)["name"],
							std::strtoul((*resultIter)["elapsed"].get<std::string>().c_str(), NULL, 0),
							std::strtoul((*resultIter)["stdDev"].get<std::string>().c_str(), NULL, 0));
							
					test->pushResults(newResult);
				}
			
				test->print();
				
				testResults.push_back(test);
				
			}
		}
	}
	
	~JsonFormat() {
		testResults.clear();
	}

	
};
