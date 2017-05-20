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
#include "../utilities/UMEBitmap.h"

#include "../utilities/ttmath/ttmath/ttmath.h"

template<typename FLOAT_T>
class HTTest : public Test {
protected:
    typedef ttmath::Big<2, 2> BigFloat;
    uint32_t *mHistogram;
    uint32_t mThreshold;
    std::string mInputFileName;
    std::string mResultFileName;

    UME::Bitmap *mInputBitmap;
    UME::Bitmap *mResultBitmap;

    struct HTLinearSolution {
        FLOAT_T r, theta; // line coordinates
        uint32_t value;   // value at coordinates
    };

    // List of solution points
    std::list<HTLinearSolution> solutions;

public:
    HTTest(bool test_enabled, std::string inputFileName, std::string resultFileName) :
        Test(test_enabled),
        mInputFileName(inputFileName),
        mResultFileName(resultFileName) {}

    UME_NEVER_INLINE virtual void initialize() {
        mInputBitmap = new UME::Bitmap(mInputFileName);
        mResultBitmap = new UME::Bitmap(mResultFileName);

        uint32_t size = mInputBitmap->GetHeight() * mInputBitmap->GetWidth();
        mHistogram = (uint32_t *) UME::DynamicMemory::AlignedMalloc(size*sizeof(uint32_t), 64);

        // Initialize arrays with random data
        for(uint32_t i = 0; i < size; i++)
        {
            mHistogram[i] = 0;
        }
        mThreshold = 0;
    }

    UME_NEVER_INLINE virtual void benchmarked_code() = 0;

    UME_NEVER_INLINE virtual void cleanup() {
        UME::DynamicMemory::AlignedFree(mHistogram);
        delete mInputBitmap;
        delete mResultBitmap;
    }

    UME_NEVER_INLINE virtual void verify() {
        uint32_t width = this->mInputBitmap->GetWidth();
        uint32_t height = this->mInputBitmap->GetHeight();
        UME::Bitmap accu_bmp(width, height, UME::PIXEL_TYPE::PIXEL_TYPE_RGB);

        uint32_t maxValue = 0;
        for (uint32_t i = 0; i < height; i++) {
            for (uint32_t j = 0; j < width; j++) {
                // Find max value
                if (mHistogram[i*width + j] > maxValue) maxValue = mHistogram[i*width + j];
            }
        }

        for (uint32_t i = 0; i < height; i++) {
            for (uint32_t j = 0; j < width; j++) {
                // Find max value
                double scaledValue = 0;
                
                if (mHistogram[i*width + j] >= mThreshold) {
                    scaledValue = (double)mHistogram[i*width + j] / (double)maxValue;
                }

                accu_bmp.SetPixelValue(j, i, uint32_t(scaledValue * (double)0xFF) << 8);
            }
        }

        accu_bmp.SaveToFile("TempResult" + get_test_identifier() + ".bmp");

        for (uint32_t i = 0; i < height; i++) {
            for (uint32_t j = 0; j < width; j++) {
                // Find max value
                double scaledValue = scaledValue = (double)mHistogram[i*width + j] / (double)maxValue;

                accu_bmp.SetPixelValue(j, i, uint32_t(scaledValue * (double)0xFF) << 8);
            }
        }
        accu_bmp.SaveToFile("TempResult_thresholded" + get_test_identifier() + ".bmp");

        /*       BigFloat sum = 0.0f;
        BigFloat avg = 0.0f;

        for(int i = 0; i < problem_size; i++)
        {
            sum += x[i];
        }

        avg = sum/(FLOAT_T)problem_size;

        error_norm_bignum = ttmath::Abs(avg - BigFloat(calculated_average));*/
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier() = 0;
};
