#include <umevector/UMEVector.h>

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"

#include "../utilities/ttmath/ttmath/ttmath.h"

template<typename FLOAT_T>
class PrototypeTest : public Test {
private:
    FLOAT_T *result;
    FLOAT_T *t0;

    int problem_size;
    static const int OPTIMAL_ALIGNMENT = 64;

public:
    PrototypeTest(int problem_size) : Test(true), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {
    result=(FLOAT_T*)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        t0=(FLOAT_T*)UME::DynamicMemory::AlignedMalloc(problem_size * sizeof(FLOAT_T), OPTIMAL_ALIGNMENT);
        for (int i=0; i < problem_size;i++)
        {
            t0[i]=static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        }
    }

    UME_NEVER_INLINE virtual void benchmarked_code() {
        UME::VECTOR::Vector<FLOAT_T> result_vec(problem_size, result);
        UME::VECTOR::Vector<FLOAT_T> v0(problem_size, t0);
        result_vec=(v0).sqrt();
    }

    UME_NEVER_INLINE virtual void cleanup() {
            UME::DynamicMemory::AlignedFree(t0);
    }

    UME_NEVER_INLINE virtual void verify() {
        //TODO 
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        return std::string("V SQRT");
    }
};
