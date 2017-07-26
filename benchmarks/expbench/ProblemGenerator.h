#pragma once

#include<random>
#include<string>

enum OP_CLASS_ID {
  OP_CLASS_SCALAR = 0,
  OP_CLASS_VECTOR,
  OP_CLASS_UNARY,
  OP_CLASS_BINARY,
  OP_CLASS_COUNT  
};

enum OP_UNARY_ID {
  OP_UNARY_SIN = 0,
  OP_UNARY_COS,
  OP_UNARY_EXP,
  OP_UNARY_LOG,
  //OP_UNARY_SQR,
  OP_UNARY_SQRT,
  OP_UNARY_COUNT
};

enum OP_BINARY_ID {
  OP_BINARY_ADD = 0,
  OP_BINARY_SUB,
  OP_BINARY_MUL,
  OP_BINARY_DIV,
  OP_BINARY_COUNT
};

class ProblemNode{
public:
    int opClass;
    int opId;
    ProblemNode* left;
    ProblemNode* right;

    ProblemNode(OP_CLASS_ID opClass, int opId, ProblemNode* left, ProblemNode* right) {
        assert(0 <= opClass && opClass < OP_CLASS_COUNT);
        
        // Check if specific classes meet the requirements
        if(opClass == OP_CLASS_SCALAR || opClass == OP_CLASS_VECTOR) {
            assert(left == nullptr);
            assert(right == nullptr);
        }
        else if(opClass == OP_CLASS_UNARY) {
            assert(0 <= opId && opId < OP_UNARY_COUNT);
            assert(left != nullptr);
            assert(right == nullptr);
        }
        else if(opClass == OP_CLASS_BINARY) {
            assert(0 <= opId && opId < OP_BINARY_COUNT);
            assert(left != nullptr);
            assert(right != nullptr);
        }

        this->opClass = opClass;
        this->opId = opId;
        this->left = left;
        this->right = right;
    }
    
    ~ProblemNode() {
        delete left;
        delete right;
    }
    
    std::string getDescriptor() {
        std::string desc = "";
        if(left != nullptr) {
            desc += left->getDescriptor();
            desc += " ";
        }
        if(right != nullptr) {
            desc += right->getDescriptor();
            desc += " ";
        }
        
        if(opClass == OP_CLASS_SCALAR) {
            desc += "S";
        }
        else if (opClass == OP_CLASS_VECTOR) {
            desc += "V";
        }
        else if (opClass == OP_CLASS_UNARY) {
            if(opId == OP_UNARY_SIN) {
                desc += "SIN";
            }
            else if(opId == OP_UNARY_COS) {
                desc += "COS";
            }
            else if(opId == OP_UNARY_EXP) {
                desc += "EXP";
            }
            else if(opId == OP_UNARY_LOG) {
                desc += "LOG";
            }
    //        else if(opId == OP_UNARY_SQR) {
    //            desc += "SQR";
    //        }
            else if(opId == OP_UNARY_SQRT) {
                desc += "SQRT";
            }
        }
        else if (opClass == OP_CLASS_BINARY) {
            if(opId == OP_BINARY_ADD) {
                desc += "ADD";
            }
            else if (opId == OP_BINARY_SUB) {
                desc += "SUB";
            }
            else if (opId == OP_BINARY_MUL) {
                desc += "MUL";
            }
            else if (opId == OP_BINARY_DIV) {
                desc += "DIV";
            }
        }

        return desc;
    }

    void print(int indent) {
        std::cout << std::endl;
        for(int i = 0; i < indent; i++) {
            std::cout << " ";
        }
        if(opClass == OP_CLASS_SCALAR) {
            std::cout << "S";
        }
        else if (opClass == OP_CLASS_VECTOR) {
            std::cout << "V";
        }
        else if (opClass == OP_CLASS_UNARY) {
            if(opId == OP_UNARY_SIN) {
                std::cout << "SIN";
            }
            else if(opId == OP_UNARY_COS) {
                std::cout << "COS";
            }
            else if(opId == OP_UNARY_EXP) {
                std::cout << "EXP";
            }
            else if(opId == OP_UNARY_LOG) {
                std::cout << "LOG";
            }
    //        else if(opId == OP_UNARY_SQR) {
    //            std::cout << "SQR";
    //        }
            else if(opId == OP_UNARY_SQRT) {
                std::cout << "SQRT";
            }
            this->left->print(indent+1);
        }
        else if (opClass == OP_CLASS_BINARY) {
            if(opId == OP_BINARY_ADD) {
                std::cout << "ADD";
            }
            else if (opId == OP_BINARY_SUB) {
                std::cout << "SUB";
            }
            else if (opId == OP_BINARY_MUL) {
                std::cout << "MUL";
            }
            else if (opId == OP_BINARY_DIV) {
                std::cout << "DIV";
            }
            this->left->print(indent+1);
            this->right->print(indent+1);
        }
    }
};

class ProblemTree {
private:
    std::random_device rd;
    int MAX_NESTING = 10;
	int NODE_COUNT_LIMIT = 20;

	int getRandomOpClass(int nesting, int nodeCount) {
		int max_value = 100;		
		if(nesting >= MAX_NESTING || nodeCount >= NODE_COUNT_LIMIT ) {
			max_value = 30;
		}
		
		std::uniform_int_distribution<int> dist(0, max_value);
		
		int opClass;
		
        int draw = dist(rd);
		if(draw <= 10) opClass = OP_CLASS_SCALAR;
		else if(draw <= 20) opClass = OP_CLASS_VECTOR;
		else if(draw <= 60) opClass = OP_CLASS_UNARY;
		else if(draw <= 100) opClass = OP_CLASS_BINARY;
		
		assert(opClass >= 0 && opClass < OP_CLASS_COUNT);
		
		return opClass;
	}
	
	
    ProblemNode* getRandomTree(int nesting, int & nodeCount) {		
        ProblemNode* newNode = nullptr;
		nodeCount++;
		
        int opClass = getRandomOpClass(nesting, nodeCount);
        assert(opClass < OP_CLASS_COUNT);
        
        if(opClass == OP_CLASS_SCALAR) {
            newNode = new ProblemNode(OP_CLASS_ID(opClass), 0, nullptr, nullptr);
        }
        else if(opClass == OP_CLASS_VECTOR) {
            newNode = new ProblemNode(OP_CLASS_ID(opClass), 0, nullptr, nullptr);
        }
        else if(opClass == OP_CLASS_UNARY) {
            std::uniform_int_distribution<int> dist2(0, OP_UNARY_COUNT-1);
            int opId = dist2(rd);
            ProblemNode* randomChild = getRandomTree(nesting+1, nodeCount);
            newNode = new ProblemNode(OP_CLASS_ID(opClass), opId, randomChild, nullptr);
        }
        else if(opClass == OP_CLASS_BINARY) {
            std::uniform_int_distribution<int> dist3(0, OP_BINARY_COUNT-1);
            int opId = dist3(rd);
            ProblemNode* randomChild1 = getRandomTree(nesting+1, nodeCount);
            ProblemNode* randomChild2 = getRandomTree(nesting+1, nodeCount);
            newNode = new ProblemNode(OP_CLASS_ID(opClass), opId, randomChild1, randomChild2);
        }
        
        return newNode;
    }
    
    std::list<OP_CLASS_ID> findLeftToRightTerminals(ProblemNode * node) {
        std::list<OP_CLASS_ID> terminals;
        
        if(node->opClass == OP_CLASS_SCALAR || node->opClass == OP_CLASS_VECTOR) {
            terminals.push_back(OP_CLASS_ID(node->opClass));
        }
        else if(node->opClass == OP_CLASS_UNARY) {
            terminals = findLeftToRightTerminals(node->left);
        }
        else if(node->opClass == OP_CLASS_BINARY) {
            terminals = findLeftToRightTerminals(node->left);
            std::list<OP_CLASS_ID> rightTerminals = findLeftToRightTerminals(node->right);
            terminals.insert(terminals.end(), rightTerminals.begin(), rightTerminals.end());
        }
        
        return terminals;
    }
    
    std::string getExpressionCode(ProblemNode * parent, int & nextTerminalId) {
        std::string code = "";

        if(parent->opClass == OP_CLASS_SCALAR) {
            code += "s" + std::to_string(nextTerminalId);
            nextTerminalId++;
        }
        else if (parent->opClass == OP_CLASS_VECTOR) {
            code += "v" + std::to_string(nextTerminalId);
            nextTerminalId++;
        }
        else if (parent->opClass == OP_CLASS_UNARY) {
            code += "(" + getExpressionCode(parent->left, nextTerminalId) + ").";
            if(parent->opId == OP_UNARY_SIN) {
                code += "sin()";
            }
            else if(parent->opId == OP_UNARY_COS) {
                code += "cos()";
            }
            else if(parent->opId == OP_UNARY_EXP) {
                code += "exp()";
            }
            else if(parent->opId == OP_UNARY_LOG) {
                code += "log()";
            }
    //        else if(parent->opId == OP_UNARY_SQR) {
    //            code += "sqr()";
    //        }
            else if(parent->opId == OP_UNARY_SQRT) {
                code += "sqrt()";
            }
        }
        else if (parent->opClass == OP_CLASS_BINARY) {
            code += "(" + getExpressionCode(parent->left, nextTerminalId) + ").";
            if(parent->opId == OP_BINARY_ADD) {
                code += "add(";
            }
            else if(parent->opId == OP_BINARY_SUB) {
                code += "sub(";
            }
            else if(parent->opId == OP_BINARY_MUL) {
                code += "mul(";
            }
            else if(parent->opId == OP_BINARY_DIV) {
                code += "div(";
            }
            code += getExpressionCode(parent->right, nextTerminalId) + ")";
        }
        
        return code;
    }
    
public:
    ProblemNode* root;
    
    ProblemTree() {
		int nodeCount = 0;
        root = getRandomTree(0, nodeCount);
    }

    ~ProblemTree() {
        delete root;
    }
    
    
    std::string getDescriptor() {
        return root->getDescriptor();
    }
    
    void print() {
        root->print(0);
    }
    
    std::list<OP_CLASS_ID> getTerminalTypes() {
        return findLeftToRightTerminals(root);
    }
    
    std::string getExpressionCode() {
        int currTerminalId = 0;
        return getExpressionCode(root, currTerminalId);
    }
};

class ProblemGenerator {    
public:
    
    ProblemTree * getRandomProblem() {
        return new ProblemTree();
    }
};
