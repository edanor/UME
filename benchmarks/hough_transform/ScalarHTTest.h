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

template<typename FLOAT_T>
class ScalarHTTest : public HTTest<FLOAT_T> {
private:

public:
    ScalarHTTest(std::string inputFileName, std::string resultFileName) : 
        HTTest<FLOAT_T>(true, inputFileName, resultFileName) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        uint32_t pixelCount = this->mInputBitmap->GetPixelCount();
        uint8_t pixelSize = this->mInputBitmap->GetPixelSize();
        uint8_t* pixel = this->mInputBitmap->GetRasterData();

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
        FLOAT_T *cos_lookup = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*thetaWidth, 64);
        FLOAT_T *sin_lookup = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*thetaWidth, 64);
        FLOAT_T theta = 0.0;
        for (uint32_t x = 0; x < thetaWidth; x++) {
            cos_lookup[x] = std::cos(theta);
            sin_lookup[x] = sin(theta);
            theta += dTheta;
        }

        for (uint32_t y = 0; y < rHeight; y++) {
            for (uint32_t x = 0; x < thetaWidth; x++) {
                uint32_t pixelValue = this->mInputBitmap->GetPixelValue(x, y);

                if (pixelValue == 0) {
                    for (uint32_t theta_coord = 0; theta_coord < thetaWidth; theta_coord++)
                    {
                        FLOAT_T r = FLOAT_T(x) * cos_lookup[theta_coord] + FLOAT_T(y)*sin_lookup[theta_coord];
                        uint32_t r_coord = uint32_t(FLOAT_T(rHeight) * ((r + R_MAX) / (R_MAX - R_MIN)));

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

        FLOAT_T* p = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*L, 64);
        memset((void*)p, 0, sizeof(FLOAT_T)*L);

        // Build the Otsu histogram
        for (uint32_t i = 0; i < N; i++) {
            uint32_t value = this->mHistogram[i];
            p[value]+= FLOAT_T(1.0);
        }

        FLOAT_T sum = 0;
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
                uk += FLOAT_T(i+1)*p[i];
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
        retval += "HT Lines Scalar, " + ScalarToString<FLOAT_T>::value();
        return retval;
    }
};
