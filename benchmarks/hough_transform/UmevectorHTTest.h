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

#include <umevector/evaluators/DyadicEvaluator.h>

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

        // Populate lookup table
        FLOAT_T *cos_lookup = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*thetaWidth, 64);
        FLOAT_T *sin_lookup = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*thetaWidth, 64);

        FLOAT_T rCoordConstMultiplier(FLOAT_T(rHeight) / (R_MAX - R_MIN));
        Vector<float> cos_vec(thetaWidth, cos_lookup);
        Vector<float> sin_vec(thetaWidth, sin_lookup);

        Vector<float> theta_vec(thetaWidth, 0.0, dTheta); // This is a linspace vector of all 'theta' values used in the algorithm.
        Vector<uint32_t> thetaOffset_int_vec(thetaWidth, uint32_t(0), uint32_t(1)); // This is a linspace vector for indices of theta values

        Vector<uint32_t> histogram(thetaWidth*rHeight, this->mHistogram); // This is a binding to the histogram (a bitmap)

        // Populate the lookup tables
        cos_vec = theta_vec.cos();
        sin_vec = theta_vec.sin();

        for (uint32_t y = 0; y < rHeight; y++) {
            for (uint32_t x = 0; x < thetaWidth; x ++) {
                uint32_t pixelValue = this->mInputBitmap->GetPixelValue(x, y);

                // Only proceed with calculations if any of the pixels is non-zero
                if (pixelValue == 0) {
                    auto sin_y_exp = sin_vec*FLOAT_T(y);
                    auto r_exp = cos_vec.fmuladd(FLOAT_T(x), sin_y_exp); // r = cos[x]*x + sin[y]*y

                    auto r_coord_exp = (rCoordConstMultiplier*(r_exp + R_MAX)).ftou(); // r scaled to bitmap dimensions

                    auto targetCoord_exp = r_coord_exp*thetaWidth + thetaOffset_int_vec; // offset in the bitmap
                    auto currValues_exp = histogram.gather(targetCoord_exp) + uint32_t(1); // gather from histogram and increment

                    MonadicEvaluator eval(histogram, currValues_exp, targetCoord_exp); // evaluate with scatter indexing
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

        auto histmax_exp = histogram.hmax();
        MonadicEvaluator eval2(&L, histmax_exp);
        L++; // There are L+1 levels including level '0'

        //std::cout << "L: " << L << std::endl;

        FLOAT_T* p = (FLOAT_T*)UME::DynamicMemory::AlignedMalloc(sizeof(FLOAT_T)*L, 64);
        memset((void*)p, 0, sizeof(FLOAT_T)*L);

        // 2.1 Build the Otsu histogram

        // Note: this loop is unlikely to be vectorized.
        //       Even if a conflict can be detected, handling
        //       it will take more time than serialize
        //       increments using scalar instructions.
        for (uint32_t i = 0; i < N; i++) {
            uint32_t value = this->mHistogram[i];
            p[value] += FLOAT_T(1.0);
        }

        // calculate 'uT'
        Vector<FLOAT_T> p_vec(L, p); // bind vector to 'p'
        FLOAT_T uT = 0;
        Vector<FLOAT_T> L_linspace_vec(L, FLOAT_T(0.0), FLOAT_T(1.0));

        auto p_normalized_exp = p_vec / FLOAT_T(N);               // p[i]/N
        auto uT_exp = (p_normalized_exp * L_linspace_vec).hadd(); // (p[i]/N)*i
        DyadicEvaluator(p_vec, p_normalized_exp, &uT, uT_exp);

        FLOAT_T maxSigmaB_sq = 0;
        for (uint32_t k = 0; k < L; k++) {
            FLOAT_T w0 = 0, w1 = 0;
            FLOAT_T uk = 0, u0 = 0, u1 = 0;

            Vector<FLOAT_T> k_linspace_vec(k+1, FLOAT_T(1.0), FLOAT_T(1.0));
            Vector<FLOAT_T> p_vec2(k+1, p); // p[0:k]
            auto w0_exp = p_vec2.hadd();
            auto uk_exp = (p_vec2*k_linspace_vec).hadd();
            DyadicEvaluator eval5(&w0, w0_exp, &uk, uk_exp);

            //std::cout << "w0: " << w0 << " uk: " << uk << "uT: " << uT << std::endl;
            w1 = 1 - w0;
            u0 = uk / w0;
            u1 = (uT - uk) / (1 - w0);
            
            // Following should hold:
            // w0*u0 + w1*u1 = uT
            // w0 + w1 = 1
            //std::cout << " Verification:\n"
            //<< " w0*u0 + w1*u1= " << w0*u0 + w1*u1 << " uT=" << uT << "\n"
            //<< " w0 + w1= " << w0 + w1 << "\n";
            assert(w0*u0 + w1*u1 <= uT*1.05);
            assert(w0*u0 + w1*u1 >= uT*0.95);
            assert(w0 + w1 <= 1.05);
            assert(w0 + w1 >= 0.95);
            //assert(false);
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
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() {
        std::string retval = "";
        retval += "HT Lines UME VECTOR, " + ScalarToString<FLOAT_T>::value();
        return retval;
    }
};
