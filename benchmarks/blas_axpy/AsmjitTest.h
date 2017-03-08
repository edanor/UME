// The MIT License (MIT)
//
// Copyright (c) 2016 CERN
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


// EXPLAINER:
// This benchmark is a test using JIT assembly instead of UME::SIMD as a 
// code generation layer. This approach requires a custom evaluator to handle 
// specific expressions. For the sake of simplicity, we are providing only a minimum
// viable evaluator. If this approach is proven to be robust and fast enough,
// a full implementation might be considered as a part of the toolkit.

#pragma once

#include <umevector/UMEVector.h>
#include "../utilities/MeasurementHarness.h"
#include "../utilities/asmjit/src/asmjit/asmjit.h"

#include "AxpyTest.h"

#define USE_ASMJIT

#if defined (USE_ASMJIT)
template<typename FLOAT_T>
class UMEVectorAsmjitSingleTest : public AxpySingleTest<FLOAT_T>{
private:
    bool isCodeGenerated;

    // JIT structures
    asmjit::JitRuntime runtime;
    asmjit::CodeHolder code;

    typedef void(*Func)(int N, float alpha, float* x, float* y);
    Func fn;

public:

    UMEVectorAsmjitSingleTest(int problem_size) : AxpySingleTest<FLOAT_T>(problem_size), isCodeGenerated(false) {
        code.init(runtime.getCodeInfo());

        if (!isCodeGenerated)
        {
            asmjit::Error err;

            asmjit::X86Compiler cc(&code);

            //cc.addFunc(asmjit::FuncSignature2<void, float*, float*>());
            cc.addFunc(asmjit::FuncSignature4<void, int, float, float*, float*>());

            ////////////////////////////////////////////
            // Declare registers
            asmjit::X86Gp cnt = cc.newInt32("cnt");
            asmjit::X86Gp x_off_reg = cc.newIntPtr("x_off_reg");
            asmjit::X86Gp y_off_reg = cc.newIntPtr("y_off_reg");
            asmjit::X86Ymm result = cc.newYmmPs("result");
            asmjit::X86Xmm alpha = cc.newXmmPs("alpha");
            asmjit::X86Ymm alpha_vec = cc.newYmmPs("alpha_vec");

            asmjit::X86Ymm t0 = cc.newYmmPs("t0");

            //asmjit::X86Mem alpha = cc.newFloatConst(asmjit::kConstScopeLocal, this->alpha);
            //asmjit::X86Mem N = cc.newInt32Const(asmjit::kConstScopeLocal, this->problem_size);

            // Define input mapping
            cc.setArg(0, cnt);
            cc.setArg(1, alpha);

            cc.setArg(2, x_off_reg);
            cc.setArg(3, y_off_reg);

            // Code generation

            asmjit::Label peel_loop_begin = cc.newLabel();
            asmjit::Label peel_loop_end = cc.newLabel();
            asmjit::Label reminder_loop_begin = cc.newLabel();
            asmjit::Label reminder_loop_end = cc.newLabel();
            asmjit::Label exit = cc.newLabel();

            err = cc.cmp(cnt, 8);
            err = cc.jl(peel_loop_end); // skip the peel loop if element count too small

            err = err = cc.vshufps(alpha_vec, alpha.ymm(), alpha.ymm(), 0); //cc.vbroadcastss(alpha_vec, alpha);

            err = cc.bind(peel_loop_begin);
                err = cc.vmulps(t0, alpha_vec, asmjit::x86::ptr(x_off_reg));
                err = cc.vaddps(result, t0, asmjit::x86::ptr(y_off_reg));
                err = cc.vmovaps(asmjit::x86::yword_ptr(y_off_reg), result.zmm());
                
                err = cc.add(x_off_reg, 32);
                err = cc.add(y_off_reg, 32);

                err = cc.add(cnt, -8);
            err = cc.cmp(cnt, 8);
            err = cc.jge(peel_loop_begin);
            err = cc.bind(peel_loop_end);

            // Check if reminder present
            err = cc.test(cnt, cnt);
            err = cc.jz(exit);  // Exit if no reminder

            // Scalar code to handle reminder
            err = cc.bind(reminder_loop_begin);
                err = cc.movss(result.xmm(), alpha.xmm());
                err = cc.mulss(result.xmm(), asmjit::x86::ptr(x_off_reg)); // multiply 'alpha' and x[i]
                err = cc.addss(result.xmm(), asmjit::x86::ptr(y_off_reg)); // add 'x*alpha' and 'y[i]'
                err = cc.movss(asmjit::x86::ptr(y_off_reg), result.xmm());

                err = cc.add(x_off_reg, 4);                          // Increment 'arr' pointer.
                err = cc.add(y_off_reg, 4);                          // Increment 'arr' pointer.

                err = cc.dec(cnt);
            err = cc.jnz(reminder_loop_begin);

            err = cc.bind(exit);
            cc.ret();

            /*err = cc.bind(loop); // start of 'for (int i = problem_size; i >= 0; i--)'

            // Loop content
            err = cc.movss(result, alpha);
            err = cc.mulss(result, asmjit::x86::ptr(x_off_reg)); // multiply 'alpha' and x[i]
            err = cc.addss(result, asmjit::x86::ptr(y_off_reg)); // add 'x*alpha' and 'y[i]'
            err = cc.movss(asmjit::x86::ptr(y_off_reg), result);

            err = cc.add(x_off_reg, 4);                          // Increment 'arr' pointer.
            err = cc.add(y_off_reg, 4);                          // Increment 'arr' pointer.

            err = cc.dec(cnt);  // i--;
            err = cc.jnz(loop); // end of 'for(int i = problem_size; i >= 0; i--)'
            err = cc.bind(exit);
            cc.ret();

            */
            ///////
            cc.endFunc();
            err = cc.finalize();

            err = runtime.add(&fn, &code);
            if (err) {
                assert(false);
            }
        }
    }

    ~UMEVectorAsmjitSingleTest() {
        runtime.release(fn);
    }


    UME_NEVER_INLINE virtual void benchmarked_code()
    {

        fn(this->problem_size, this->alpha, this->x, this->y);
        //UME::VECTOR::Vector<FLOAT_T> x_vec(this->problem_size, this->x);
        //UME::VECTOR::Vector<FLOAT_T> y_vec(this->problem_size, this->y);
        //UME::VECTOR::Scalar<FLOAT_T> alpha_s(this->alpha);

        //y_vec = alpha_s*x_vec + y_vec;

    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "Asmjit single, (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(this->problem_size);
        return retval;
    }
};

template<typename FLOAT_T>
class UMEVectorAsmjitChainedTest : public AxpyChainedTest<FLOAT_T> {
public:
    UMEVectorAsmjitChainedTest(int problem_size) : AxpyChainedTest<FLOAT_T>(problem_size) {}

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        UME::VECTOR::Vector<FLOAT_T> x0_vec(this->problem_size, this->x0);
        UME::VECTOR::Vector<FLOAT_T> x1_vec(this->problem_size, this->x1);
        UME::VECTOR::Vector<FLOAT_T> x2_vec(this->problem_size, this->x2);
        UME::VECTOR::Vector<FLOAT_T> x3_vec(this->problem_size, this->x3);
        UME::VECTOR::Vector<FLOAT_T> x4_vec(this->problem_size, this->x4);
        UME::VECTOR::Vector<FLOAT_T> x5_vec(this->problem_size, this->x5);
        UME::VECTOR::Vector<FLOAT_T> x6_vec(this->problem_size, this->x6);
        UME::VECTOR::Vector<FLOAT_T> x7_vec(this->problem_size, this->x7);
        UME::VECTOR::Vector<FLOAT_T> x8_vec(this->problem_size, this->x8);
        UME::VECTOR::Vector<FLOAT_T> x9_vec(this->problem_size, this->x9);
        UME::VECTOR::Vector<FLOAT_T> y_vec(this->problem_size, this->y);

        UME::VECTOR::Scalar<FLOAT_T> alpha0(this->alpha[0]);
        UME::VECTOR::Scalar<FLOAT_T> alpha1(this->alpha[1]);
        UME::VECTOR::Scalar<FLOAT_T> alpha2(this->alpha[2]);
        UME::VECTOR::Scalar<FLOAT_T> alpha3(this->alpha[3]);
        UME::VECTOR::Scalar<FLOAT_T> alpha4(this->alpha[4]);
        UME::VECTOR::Scalar<FLOAT_T> alpha5(this->alpha[5]);
        UME::VECTOR::Scalar<FLOAT_T> alpha6(this->alpha[6]);
        UME::VECTOR::Scalar<FLOAT_T> alpha7(this->alpha[7]);
        UME::VECTOR::Scalar<FLOAT_T> alpha8(this->alpha[8]);
        UME::VECTOR::Scalar<FLOAT_T> alpha9(this->alpha[9]);

        y_vec = y_vec + alpha0*x0_vec + alpha1*x1_vec +
            alpha2*x2_vec + alpha3*x3_vec +
            alpha4*x4_vec + alpha5*x5_vec +
            alpha6*x6_vec + alpha7*x7_vec +
            alpha8*x8_vec + alpha9*x9_vec;
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "UME::VECTOR chained, (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(this->problem_size);
        return retval;
    }
};

#else

template<typename FLOAT_T>
class UMEVectorAsmjitSingleTest : public Test {
public:
    int problem_size;

    UMEVectorAsmjitSingleTest(int problem_size) : Test(false), problem_size(problem_size) {}
    ~UMEVectorAsmjitSingleTest() {}

    UME_NEVER_INLINE virtual void initialize() {}
    UME_NEVER_INLINE virtual void benchmarked_code() {}
    UME_NEVER_INLINE virtual void cleanup() {}
    UME_NEVER_INLINE virtual void verify() {}
    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "Asmjit single, (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(this->problem_size);
        return retval;
    }
};

template<typename FLOAT_T>
class UMEVectorAsmjitChainedTest : public Test {
public:
    int problem_size;

    UMEVectorAsmjitChainedTest(int problem_size) : Test(false), problem_size(problem_size) {}

    UME_NEVER_INLINE virtual void initialize() {}
    UME_NEVER_INLINE virtual void benchmarked_code() {}
    UME_NEVER_INLINE virtual void cleanup() {}
    UME_NEVER_INLINE virtual void verify() {}
    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "Asmjit chained, (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(this->problem_size);
        return retval;
    }
};


#endif