// 
// This piece of code comes from https://github.com/dpiparo/vdt.
// This code is not a part of UME::SIMD library code and is used purely for
// performance measurement reference.
// 
// Modifications have been made to original files to fit them for benchmarking
// of UME::SIMD.

#ifndef EXPLOG_COMMON_H_
#define EXPLOG_COMMON_H_

#include <cmath>
#include <limits>
#include <cassert>

#include "vdtcore_common.h"
#include "vdt_exp.h"
#include "vdt_log.h"

template<typename SCALAR_FLOAT_T>
inline SCALAR_FLOAT_T call_exp_vdt(SCALAR_FLOAT_T in) {
    (void)in; // ignore unused parameter, do nothing
    assert(0);
}

template<>
inline float call_exp_vdt<float>(float in) {
    return vdt::fast_expf(in);
}

template<>
inline double call_exp_vdt<double>(double in) {
    return vdt::fast_exp(in);
}

template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_exp_vdt_scalar(int ARRAY_SIZE)
{
    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T* input = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* output = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* values = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);

    generate_some_exp_values<SCALAR_FLOAT_T>(LEN, input, output);
    
    start = get_timestamp();
    
    for(int i = 0; i < LEN; i++) {
        values[i] = call_exp_vdt<SCALAR_FLOAT_T>(input[i]);
    }

    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
        SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

        if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
        //if (output_sin[i] != values_sin[i])
        //    std::cout << " Difference in sin[" << i << "]: " << values_sin[i]
        //   << " should be: " << output_sin[i]
        //    << " error(ulp): " << error_ulp << std::endl;
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;

    UME::DynamicMemory::AlignedFree(input);
    UME::DynamicMemory::AlignedFree(output);
    UME::DynamicMemory::AlignedFree(values);

    return result;
}

template<typename SCALAR_FLOAT_T>
UME_FORCE_INLINE SCALAR_FLOAT_T call_log_vdt(SCALAR_FLOAT_T in) {
    (void)in; // ignore unused parameter, do nothing
    assert(0);
}

template<>
UME_FORCE_INLINE float call_log_vdt<float>(float in) {
    return vdt::fast_logf(in);
}

template<>
UME_FORCE_INLINE double call_log_vdt<double>(double in) {
    return vdt::fast_log(in);
}

UME_FORCE_INLINE float call_log2_vdt(float in) {
    float inv_log_of_2 = 1.4426950408889634073599246810019f; // 1/log(2)
    return inv_log_of_2 * vdt::fast_logf(in);
}

UME_FORCE_INLINE double call_log2_vdt(double in) {
    double inv_log_of_2 = 1.4426950408889634073599246810019; // 1/log(2)
    return inv_log_of_2 * vdt::fast_log(in);
}

UME_FORCE_INLINE float call_log10_vdt(float in) {
    float inv_log_of_10 = 0.4342944819032518276511289189166f; // 1/log(10)
    return inv_log_of_10 * vdt::fast_logf(in);
}

UME_FORCE_INLINE double call_log10_vdt(double in) {
    double inv_log_of_10 = 0.4342944819032518276511289189166f; // 1/log(10)
    return inv_log_of_10 * vdt::fast_log(in);
}

template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_log_vdt_scalar(int ARRAY_SIZE)
{
    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T* input = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* output = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* values = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);

    generate_some_log_values<SCALAR_FLOAT_T>(LEN, input, output);
    
    start = get_timestamp();
    
    for(int i = 0; i < LEN; i++) {
        values[i] = call_log_vdt<SCALAR_FLOAT_T>(input[i]);
    }

    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
        SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

        if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
        //if (output_sin[i] != values_sin[i])
        //    std::cout << " Difference in sin[" << i << "]: " << values_sin[i]
        //   << " should be: " << output_sin[i]
        //    << " error(ulp): " << error_ulp << std::endl;
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;

    UME::DynamicMemory::AlignedFree(input);
    UME::DynamicMemory::AlignedFree(output);
    UME::DynamicMemory::AlignedFree(values);

    return result;
}

template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_log2_vdt_scalar(int ARRAY_SIZE)
{
    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T* input = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* output = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* values = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);

    generate_some_log_values<SCALAR_FLOAT_T>(LEN, input, output);
    
    start = get_timestamp();
    
    for(int i = 0; i < LEN; i++) {
        values[i] = call_log2_vdt(input[i]);
    }

    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
        SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

        if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
        //if (output_sin[i] != values_sin[i])
        //    std::cout << " Difference in sin[" << i << "]: " << values_sin[i]
        //   << " should be: " << output_sin[i]
        //    << " error(ulp): " << error_ulp << std::endl;
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;

    UME::DynamicMemory::AlignedFree(input);
    UME::DynamicMemory::AlignedFree(output);
    UME::DynamicMemory::AlignedFree(values);

    return result;
}

template<typename SCALAR_FLOAT_T>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_log10_vdt_scalar(int ARRAY_SIZE)
{
    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T* input = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* output = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);
    SCALAR_FLOAT_T* values = (SCALAR_FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), 64);

    generate_some_log10_values<SCALAR_FLOAT_T>(LEN, input, output);
    
    start = get_timestamp();
    
    for(int i = 0; i < LEN; i++) {
        values[i] = call_log10_vdt(input[i]);
    }

    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
        SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

        if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
        //if (output_sin[i] != values_sin[i])
        //    std::cout << " Difference in sin[" << i << "]: " << values_sin[i]
        //   << " should be: " << output_sin[i]
        //    << " error(ulp): " << error_ulp << std::endl;
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;

    UME::DynamicMemory::AlignedFree(input);
    UME::DynamicMemory::AlignedFree(output);
    UME::DynamicMemory::AlignedFree(values);

    return result;
}

#endif 
