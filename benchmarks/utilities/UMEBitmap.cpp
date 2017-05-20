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

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cstring>

#include <umesimd/UMEMemory.h>

#include "UMEBitmap.h"
#include "UMEEndianness.h"

#include <assert.h>

UME_FORCE_INLINE bool UME::operator == (UME::BitmapFileHeader & h0, UME::BitmapFileHeader & h1)
{
    if(     h0.fileSize != h1.fileSize// Check size first for better performance
        ||  h0.headerID != h1.headerID
        ||  h0.imageOffset != h1.imageOffset
        ||  h0.reserved1 != h1.reserved1
        ||  h0.reserved2 != h1.reserved2)
    {
        return false;
    }

    return true;
}


UME_FORCE_INLINE bool UME::operator != (UME::BitmapFileHeader & h0, UME::BitmapFileHeader & h1)
{
    if(     h0.fileSize != h1.fileSize // Check size first for better performance
        ||  h0.headerID != h1.headerID
        ||  h0.imageOffset != h1.imageOffset
        ||  h0.reserved1 != h1.reserved1
        ||  h0.reserved2 != h1.reserved2)
    {
        return true;
    }

    return false;
}

UME_FORCE_INLINE bool UME::operator == (UME::BitmapDIBHeader & h0, UME::BitmapDIBHeader & h1)
{
    if(     h0.headerSize != h1.headerSize // Check size first for better performance
        ||  h0.bitsPerPixel != h1.bitsPerPixel
        ||  h0.colorPlanes != h1.colorPlanes
        ||  h0.height != h1.height
        ||  h0.width != h1.width)
    {
        return true;
    }

    return false;
}

UME_FORCE_INLINE bool UME::operator != (UME::BitmapDIBHeader & h0, UME::BitmapDIBHeader & h1)
{
    if(     h0.headerSize != h1.headerSize // Check size first for better performance
        ||  h0.bitsPerPixel != h1.bitsPerPixel
        ||  h0.colorPlanes != h1.colorPlanes
        ||  h0.height != h1.height
        ||  h0.width != h1.width)
    {
        return false;
    }

    return true;
}

bool UME::Bitmap::LoadFromFile(std::string const & fileName)
{
    FILE *file = NULL;
    bool retval = true;
    size_t read_size = 0;
    
    do
    {
#if defined (_MSC_VER)
        fopen_s(&file, fileName.c_str(), "rb");
#else
        file = fopen(fileName.c_str(), "rb");
#endif
        if (!file)
        {
            std::cerr << "Error: cannot read file: " << fileName << std::endl;
            retval = false;
            break;
        }

        // Read the file header
        read_size = fread(mHeader.raw, 1, UME_BITMAP_HEADER_LENGTH, file);
        if(read_size != UME_BITMAP_HEADER_LENGTH) {
            std::cerr << "Error: reading bitmap header: " << fileName << std::endl;
        }

        // Parse the header 
        mHeader.headerID = READ_WORD(mHeader.raw);
        mHeader.fileSize = READ_DWORD(mHeader.raw + 2);
        mHeader.reserved1 = READ_WORD(mHeader.raw + 6);
        mHeader.reserved2 = READ_WORD(mHeader.raw + 8);
        mHeader.imageOffset = READ_DWORD(mHeader.raw + 10);

        // Read the DIB header

        read_size = fread(mDIBHeader.raw, 1, 4, file);
        if(read_size != 4) {
            std::cerr << "Error: reading DIB header size: " << fileName << std::endl;
        }
        mDIBHeader.headerSize = READ_DWORD(mDIBHeader.raw);

        // TODO: different types of headers can be supported. Differentiation depends on header size.
        if (mDIBHeader.headerSize != UME_BITMAP_DIB_HEADER_LENGTH)
        {
            std::cerr << "Error: invalid size of dIBHeader: " << mDIBHeader.headerSize << std::endl;
            retval = false;
            break;
        }

        read_size = fread(mDIBHeader.raw + 4, 1, mDIBHeader.headerSize - 4, file);
        if(read_size != mDIBHeader.headerSize - 4) {
            std::cerr << "Error: reading DIB header: " << fileName << std::endl;
        }

        mDIBHeader.width  = READ_DWORD(mDIBHeader.raw + 4);
        mDIBHeader.height = READ_DWORD(mDIBHeader.raw + 8);
        mDIBHeader.colorPlanes = READ_WORD(mDIBHeader.raw + 12);
        mDIBHeader.bitsPerPixel = READ_WORD(mDIBHeader.raw + 14);
        
        mPaddedWidth = (uint32_t) std::ceil((double)mDIBHeader.width*mDIBHeader.bitsPerPixel / 32)*4;

        // Read the bitmap
        unsigned int bitmapSize = GetBitmapSize();
        // Sanity check
        if(bitmapSize != mPaddedWidth*mDIBHeader.height)
        {
            std::cout << "UMEBitmap: error - invalid line padding!" << std::endl;
        }

        mRasterData = (uint8_t*)UME::DynamicMemory::Malloc(bitmapSize);
        read_size = fread(mRasterData, 1, bitmapSize, file);
        if(read_size != bitmapSize) {
            std::cerr << "Error: reading bitmap data: " << fileName << std::endl;
        }
    } while (0);

    fclose(file);
    return retval;
}
