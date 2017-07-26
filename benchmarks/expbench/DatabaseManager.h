#pragma once

#include "JsonFormat.h"
#include "../utilities/sqlite/sqlite3.h"

class DatabaseManager {
private:
	std::string dbFileName;
	
public:

	DatabaseManager(std::string const & databaseFileName) 
		: dbFileName(databaseFileName)
	{}
	
	int insert(std::list<TestDesc*> & tests) {
		sqlite3 *dbObject;
		int retval = sqlite3_open(dbFileName.c_str(), &dbObject);
		
		if(retval != 0) {
			std::cout << "Failed to open database!\n";
		}
		else {
			
			for(auto testIter = tests.begin(); testIter != tests.end(); testIter++) {
				std::string precisionStr = "";
				std::string problemSizeStr = "";
				std::string executionsStr = "";

				for(auto paramIter = (*testIter)->parameters.begin(); paramIter != (*testIter)->parameters.end(); paramIter++) {
					if((*paramIter)->name == "precision") {
						precisionStr = std::to_string((*paramIter)->value);
					}
					else if ((*paramIter)->name == "problem_size") {
						problemSizeStr = std::to_string((*paramIter)->value);
					}
					else if ((*paramIter)->name == "executions") {
						executionsStr = std::to_string((*paramIter)->value);
					}
				}

				std::string platformStr = "Unknown";
				std::string implStr = (*testIter)->name;
				
				for(auto resultIter = (*testIter)->results.begin(); resultIter != (*testIter)->results.end(); resultIter++) {
					std::string problemStr = (*resultIter)->name;
					std::string timeAvgStr = std::to_string((*resultIter)->elapsed);
					std::string timeDevStr = std::to_string((*resultIter)->stdDev);
					
					std::string sqlCmd = 
						"INSERT INTO measurements ("
							"Platform, "
							"Problem, "
							"Implementation, "
							"Precision, "
							"Executions, "
							"ProblemSize, "
							"TimeAverage, "
							"TimeDeviation) VALUES (";
							
					sqlCmd += "'" + platformStr + "', "
						   +  "'" + problemStr + "', "
						   +  "'" + implStr + "', "
						   +  precisionStr + ", "
						   +  executionsStr + ", "
						   +  problemSizeStr + ", "
						   +  timeAvgStr + ", "
						   +  timeDevStr + ");";
						   
					std::cout << sqlCmd << std::endl;
					
					sqlite3_stmt *sqlStmt;
					
					// Prepare command
					sqlite3_prepare(dbObject, sqlCmd.c_str(), -1, &sqlStmt, NULL);
					// Execute insertion
					sqlite3_step(sqlStmt);
					sqlite3_finalize(sqlStmt);
				}
						
						
			}
			
			std::cout << "Database open\n";
			sqlite3_close(dbObject);
		}
		
		return retval;
	}
	
	
	
};