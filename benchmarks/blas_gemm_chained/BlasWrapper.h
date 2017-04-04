#pragma once

#include <cblas.h>

template<typename FLOAT_T>
class GEMM_kernel {
    // Only specializations of this class should be allowed.
private:
    GEMM_kernel() {}
    ~GEMM_kernel() {}
};

template<>
class GEMM_kernel<float> {
public:
    // Assume ROW_MAJOR matrix orientation
    UME_FORCE_INLINE static void blas_gemm(int N, float alpha, float *A, float *B, float beta, float *C) {
        cblas_sgemm(
            CblasRowMajor, // Layout
            CblasNoTrans,  // Trans A
            CblasNoTrans,  // Trans B
            N,             // # of rows A (m)
            N,             // # of cols B (n)
            N,             // # of cols A, rows B (k)
            alpha,         // scalar coeff.
            A,             // matrix A
            N,             // lda
            B,             // matrix B
            N,             // ldb
            beta,          // scalar coeff.
            C,             // matrix C
            N);            // ldc
    }
};

template<>
class GEMM_kernel<double> {
public:
    // Assume ROW_MAJOR matrix orientation
    UME_FORCE_INLINE static void blas_gemm(int N, double alpha, double *A, double *B, double beta, double *C) {
        cblas_dgemm(
            CblasRowMajor, // Layout
            CblasNoTrans,  // Trans A
            CblasNoTrans,  // Trans B
            N,             // # of rows A (m)
            N,             // # of cols B (n)
            N,             // # of cols A, rows B (k)
            alpha,         // scalar coeff.
            A,             // matrix A
            N,             // lda
            B,             // matrix B
            N,             // ldb
            beta,          // scalar coeff.
            C,             // matrix C
            N);            // ldc
    }
};