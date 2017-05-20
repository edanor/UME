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
#pragma once

#include "../utilities/MeasurementHarness.h"
#include "../utilities/UMEScalarToString.h"

#include "HTTest.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
// Use UME::SIMD for portable allocators only
#include <umesimd/UMESimd.h>

using namespace UME::SIMD;

template<typename FLOAT_T>
class Avx512HTTest : public HTTest<FLOAT_T> {
public:
    Avx512HTTest(std::string inputFileName, std::string resultFileName) :
        HTTest<FLOAT_T>(false, inputFileName, resultFileName) {}
    UME_NEVER_INLINE virtual void benchmarked_code() {} // Dummy function
    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "HT Lines AVX512 " + ScalarToString<float>::value();
        return retval;
    }
};

#if defined(__AVX512F__)

template<>
class Avx512HTTest<float> : public HTTest<float> {
private:
    const int SIMD_STRIDE = 16;
public:
    Avx512HTTest(std::string inputFileName, std::string resultFileName) :
        HTTest<float>(true, inputFileName, resultFileName) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        uint32_t pixelCount = this->mInputBitmap->GetPixelCount();
        uint8_t pixelSize = this->mInputBitmap->GetPixelSize();
        uint8_t* pixel = this->mInputBitmap->GetRasterData();

        uint32_t thetaWidth = this->mInputBitmap->GetWidth();
        uint32_t rHeight = this->mInputBitmap->GetHeight();

        float dTheta = float(M_PI) / float(thetaWidth);
        const float R_MAX = float(thetaWidth) + float(rHeight);
        const float R_MIN = -R_MAX;

        // 1. ACCUMULATION (VOTING) PHASE
        // In this phase we traverse the input bitmap and
        // look for black pixels. For each black pixel we map it in
        // the transformed space, and increment required accumulator values.
        // The input to this phase is the bitmap image, the output is Hough space
        // histogram filled with votes.
        int PIXEL_SIZE = this->mInputBitmap->GetPixelSize();
        uint8_t *data = this->mInputBitmap->GetRasterData();

        // Populate a register with theta offset values
        alignas(64) uint32_t raw[SIMD_STRIDE];
        for (int i = 0; i < SIMD_STRIDE; i++) {
            raw[i] = uint32_t(i);
        }

        __m512i thetaOffset_int_vec = _mm512_load_si512((__m512i*)raw);

        const __m512 dTheta_vec = _mm512_set1_ps(dTheta);
        const __m512 rCoordConstMultiplier = _mm512_set1_ps(float(rHeight) / (R_MAX - R_MIN));
        const __m512 RMAX_vec = _mm512_set1_ps(R_MAX);

        __m512 cos_vec, sin_vec, r_vec, x_vec, y_vec;
        __m512i currValues_vec;

        // Populate lookup table
        float *cos_lookup = (float*)UME::DynamicMemory::AlignedMalloc(sizeof(float)*thetaWidth, 32);
        float *sin_lookup = (float*)UME::DynamicMemory::AlignedMalloc(sizeof(float)*thetaWidth, 32);
        float theta = 0.0;
        for (uint32_t x = 0; x < thetaWidth; x++) {
            cos_lookup[x] = std::cos(theta);
            sin_lookup[x] = std::sin(theta);
            theta += dTheta;
        }

        for (uint32_t y = 0; y < rHeight; y++) {
            for (uint32_t x = 0; x < thetaWidth; x++) {
                uint32_t pixelValue = this->mInputBitmap->GetPixelValue(x, y);

                // Only proceed with calculations if any of the pixels is non-zero
                if (pixelValue == 0) {
                    x_vec = _mm512_set1_ps(float(x));
                    y_vec = _mm512_set1_ps(float(y));
                    for (uint32_t theta_coord = 0; theta_coord < thetaWidth; theta_coord += SIMD_STRIDE) {
                        sin_vec = _mm512_load_ps(&sin_lookup[theta_coord]);
                        cos_vec = _mm512_load_ps(&cos_lookup[theta_coord]);

                        r_vec = _mm512_mul_ps(x_vec, cos_vec);
                        r_vec = _mm512_fmadd_ps(y_vec, sin_vec, r_vec);

                        __m512 t0 = _mm512_add_ps(r_vec, RMAX_vec);
                        __m512 t1 = _mm512_mul_ps(t0, rCoordConstMultiplier);
                        __m512i r_coord = _mm512_cvtps_epi32(t1);

                        __m512i t2 = _mm512_mullo_epi32(r_coord, _mm512_set1_epi32(thetaWidth));
                        __m512i t3 = _mm512_add_epi32(t2, thetaOffset_int_vec);
                        __m512i targetCoord_vec = _mm512_add_epi32(t3, _mm512_set1_epi32(theta_coord));

                        __m512i t4 = _mm512_i32gather_epi32(targetCoord_vec, this->mHistogram, 4);
                        __m512i t5 = _mm512_add_epi32(t4, _mm512_set1_epi32(1));
                        _mm512_i32scatter_epi32(this->mHistogram, targetCoord_vec, t5, 4);
                    }
                    // Handle the remainder loop
                    for (uint32_t theta_coord = (thetaWidth / SIMD_STRIDE)*SIMD_STRIDE; theta_coord < thetaWidth; theta_coord++)
                    {
                        float r = float(x) * cos_lookup[theta_coord] + float(y)*sin_lookup[theta_coord];
                        uint32_t r_coord = uint32_t(float(rHeight) * ((r + R_MAX) / (R_MAX - R_MIN)));

                        // Calculate offset of current pixel (32b aligned lanes)
                        uint32_t targetCoord = r_coord*thetaWidth + theta_coord;
                        // Increment current accumulator value
                        this->mHistogram[targetCoord]++;
                    }
                }
            }
        }

        UME::DynamicMemory::AlignedFree(sin_lookup);
        UME::DynamicMemory::AlignedFree(cos_lookup);

        // 2. SEGMENTATION PHASE
        // In this phase, we perform thresholding. Thresholding is required so that
        // we obtain separated clusters representing each solution. We will be using
        // Otsu method to find the optimal threshold.

        uint32_t L = 0;
        uint32_t N = rHeight*thetaWidth;
        for (uint32_t i = 0; i < N; i++) {
            L = std::max(L, this->mHistogram[i]);
        }
        L++; // There are L+1 levels including level '0'

             //std::cout << "L: " << L << std::endl;

        float* p = (float*)UME::DynamicMemory::AlignedMalloc(sizeof(float)*L, 32);
        memset((void*)p, 0, sizeof(float)*L);

        // Build the Otsu histogram
        for (uint32_t i = 0; i < N; i++) {
            uint32_t value = this->mHistogram[i];
            p[value] += float(1.0);
        }

        float sum = 0;
        float uT = 0;
        for (uint32_t i = 0; i < L; i++) {
            p[i] = p[i] / float(N);
            uT = float(i)*p[i];
        }

        float maxSigmaB_sq = 0;
        for (uint32_t k = 0; k < L; k++) {
            float w0 = 0, w1 = 0;
            float uk = 0, u0 = 0, u1 = 0;
            for (uint32_t i = 0; i <= k; i++) {
                w0 += p[i];
                uk += float(i + 1)*p[i];
            }
            w1 = 1 - w0;
            u0 = uk / w0;
            u1 = (uT - uk) / (1 - w0);

            // Following should hold:
            // w0*u0 + w1*u1 = uT
            // w0 + w1 = 1
            /*std::cout << " Verification:\n"
            << " w0*u0 + w1*u1= " << w0*u0 + w1*u1 << " uT=" << uT << "\n"
            << " w0 + w1= " << w0 + w1 << "\n";*/
            assert(w0*u0 + w1*u1 < uT*1.05);
            assert(w0*u0 + w1*u1 > uT*0.95);
            assert(w0 + w1 < 1.05);
            assert(w0 + w1 > 0.95);

            // calculate the objective function
            float sigmaB_sq = (uT*w0 - uk)*(uT*w0 - uk) / (w0 * (1 - w0));
            // look for MAX
            if (sigmaB_sq > maxSigmaB_sq) {
                //std::cout << " Previous: " << maxSigmaB_sq << "\n";
                //std::cout << " New: " << sigmaB_sq << "\n";
                maxSigmaB_sq = sigmaB_sq;
                this->mThreshold = k;
            }
        }
        UME::DynamicMemory::AlignedFree(p);

        //std::cout << "Threshold: " << this->mThreshold << std::endl;
        this->mThreshold *= 0.95;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "HT Lines AVX512 " + ScalarToString<float>::value();
        return retval;
    }
};

#endif //defined(__AVX512F__)
