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
#include <umevector/UMEVector.h>

using namespace UME::VECTOR;

template<typename FLOAT_T>
class UmevectorHTTest : public HTTest<FLOAT_T> {
private:

public:
    UmevectorHTTest(std::string inputFileName, std::string resultFileName) :
        HTTest<FLOAT_T>(true, inputFileName, resultFileName) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        uint32_t thetaWidth = this->mInputBitmap->GetWidth();
        uint32_t rHeight = this->mInputBitmap->GetHeight();

        FLOAT_T dTheta = FLOAT_T(M_PI) / FLOAT_T(thetaWidth);
        const FLOAT_T R_MAX = FLOAT_T(thetaWidth) + FLOAT_T(rHeight);
        const FLOAT_T R_MIN = -R_MAX;

        // 1. ACCUMULATION (VOTING) PHASE
        // In this phase we traverse the input bitmap and
        // look for black pixels. For each black pixel we map it in
        // the transformed space, and increment required accumulator values.
        // The input to this phase is the bitmap image, the output is Hough space
        // histogram filled with votes.

        // Populate a register with theta offset values
        //alignas(SIMDVec<uint32_t, SIMD_STRIDE>::alignment()) uint32_t raw[SIMD_STRIDE];
        //for (int i = 0; i < SIMD_STRIDE; i++) {
        //    raw[i] = uint32_t(i);
        //}

        //SIMDVec<uint32_t, SIMD_STRIDE> thetaOffset_int_vec(raw);
        //SIMDVec<FLOAT_T, SIMD_STRIDE> thetaOffset_vec = thetaOffset_int_vec;
        //const SIMDVec<FLOAT_T, SIMD_STRIDE> dTheta_vec(dTheta);
        FLOAT_T rCoordConstMultiplier(FLOAT_T(rHeight) / (R_MAX - R_MIN));
        //const SIMDVec<FLOAT_T, SIMD_STRIDE> RMAX_vec(R_MAX);
        //SIMDVec<FLOAT_T, SIMD_STRIDE> theta_vec;

        //SIMDVec<FLOAT_T, SIMD_STRIDE> cos_vec, sin_vec;
        //SIMDVec<uint32_t, SIMD_STRIDE> r_coord, targetCoord_vec, currValues_vec;

        // Populate lookup table
        FLOAT_T *cos_lookup = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*thetaWidth, 64);
        FLOAT_T *sin_lookup = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*thetaWidth, 64);
        //FLOAT_T theta = 0.0;

        Vector<float> cos_vec(thetaWidth, cos_lookup);
        Vector<float> sin_vec(thetaWidth, sin_lookup);

        Vector<float> theta_vec(thetaWidth, 0.0, dTheta); // This is a linspace vector
        Vector<uint32_t> thetaOffset_int_vec(thetaWidth, uint32_t(0), uint32_t(1));

        Vector<uint32_t> histogram(thetaWidth*rHeight, this->mHistogram);

        // Populate the lookup table
        cos_vec = theta_vec.cos();
        sin_vec = theta_vec.sin();

        for (uint32_t y = 0; y < rHeight; y++) {
            for (uint32_t x = 0; x < thetaWidth; x ++) {
                uint32_t pixelValue = this->mInputBitmap->GetPixelValue(x, y);

                // Only proceed with calculations if any of the pixels is non-zero
                if (pixelValue == 0) {
                    auto t0 = sin_vec*FLOAT_T(y);
                    //auto r_vec = cos_vec * FLOAT_T(x) + t0;
                    auto r_vec = cos_vec.fmuladd(FLOAT_T(x), t0);

                    auto r_coord_0 = rCoordConstMultiplier*(r_vec + R_MAX);
                    auto r_coord_1 = r_coord_0.ftou();

                    auto targetCoord_vec = r_coord_1*thetaWidth + thetaOffset_int_vec;

                    auto currValues_0 = histogram.gather(targetCoord_vec);
                    auto currValues_1 = currValues_0 + uint32_t(1);

                    MonadicEvaluator eval(histogram, currValues_1, targetCoord_vec);
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

        FLOAT_T* p = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*L, 64);
        memset((void*)p, 0, sizeof(FLOAT_T)*L);

        // Build the Otsu histogram
        for (uint32_t i = 0; i < N; i++) {
            uint32_t value = this->mHistogram[i];
            p[value] += FLOAT_T(1.0);
        }

        FLOAT_T uT = 0;
        for (uint32_t i = 0; i < L; i++) {
            p[i] = p[i] / FLOAT_T(N);
            uT = FLOAT_T(i)*p[i];
        }

        FLOAT_T maxSigmaB_sq = 0;
        for (uint32_t k = 0; k < L; k++) {
            FLOAT_T w0 = 0, w1 = 0;
            FLOAT_T uk = 0, u0 = 0, u1 = 0;
            for (uint32_t i = 0; i <= k; i++) {
                w0 += p[i];
                uk += FLOAT_T(i + 1)*p[i];
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
            FLOAT_T sigmaB_sq = (uT*w0 - uk)*(uT*w0 - uk) / (w0 * (1 - w0));
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
        retval += "HT Lines UME SIMD, " + ScalarToString<FLOAT_T>::value();
        return retval;
    }
};
