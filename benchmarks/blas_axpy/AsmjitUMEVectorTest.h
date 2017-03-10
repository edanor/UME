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

#include <algorithm>

#define USE_ASMJIT

#if defined (USE_ASMJIT)

// This is a POC for a Monadic evaluator using AsmJIT as the code-generation mechanism.
// 1. We need to traverse the tree and store pointers to all terminal symbols, so that
//    a function can be called with proper data
// 2. We need to traverse the tree and compile the function. This should happen preferrably
//    only once per expression evaluation (some global context needed?).
// 3. We have to call the pre-compiled function on local data settings.
template<uint32_t VEC_LEN, uint32_t SIMD_STRIDE>
class AsmjitEvaluator {
    asmjit::JitRuntime runtime;
    asmjit::CodeHolder code;
    asmjit::X86Compiler* cc;
    asmjit::FuncSignatureX evaluator_signature;

    static const int MAX_ARG_COUNT = 128;

    int argCount;

    typedef void(*wrapperFunc)(void);

    // Argument 1 is the destination, but since argument 0 is number of elements, we 
    // dont store it here. 'arguments[0]' is then destination pointer.
    uint64_t arguments[MAX_ARG_COUNT];
    bool isScalar[MAX_ARG_COUNT];
    // Offset registers for each pointer argument
    asmjit::X86Gp offsetRegisters[MAX_ARG_COUNT];

public:
    template<typename SCALAR_T, typename EXP_T>
    AsmjitEvaluator(
        UME::VECTOR::Vector<SCALAR_T, VEC_LEN, SIMD_STRIDE> & dst,
        UME::VECTOR::ArithmeticExpression<SCALAR_T, SIMD_STRIDE, EXP_T> & exp)
    {
        EXP_T & reinterpret_exp = static_cast<EXP_T &>(exp);

        code.init(runtime.getCodeInfo());
        asmjit::Error err;
        cc = new asmjit::X86Compiler(&code);


        // configure the function
        evaluator_signature.setRetT<void>(); // the function evaluates by side-effects on input parameters

        asmjit::X86Gp cnt = cc->newInt32("cnt");
        offsetRegisters[0] = cc->newIntPtr("Dst");

        evaluator_signature.addArg(asmjit::TypeIdOf<int>::kTypeId);
        evaluator_signature.addArg(asmjit::TypeIdOf<SCALAR_T*>::kTypeId);

        // Visit all nodes and figure out the function signature.
        argCount = 1;
        map_arguments(reinterpret_exp);

        // Now when all arguments are already mapped, we can initialize parameters
        asmjit::CCFunc* evaluator = cc->addFunc(evaluator_signature);

        // Initialize iteration count
        asmjit::X86Mem t0 = cc->newInt64Const(asmjit::kConstScopeLocal, dst.LENGTH());
        cc->mov(cnt, t0);

        // Initialize destination offset register
        asmjit::X86Mem t1 = cc->newInt64Const(asmjit::kConstScopeLocal, (uint64_t)dst.elements);
        cc->mov(offsetRegisters[0], t1);

        // Initialize remaining offset registers
        for (int i = 1; i < argCount; i++) {
            asmjit::X86Mem tX = cc->newInt64Const(asmjit::kConstScopeLocal, arguments[i]);
            cc->mov(offsetRegisters[i], tX);
        }

        {
            //err = cc->setArg(0, cnt);
            //err = cc->setArg(1, offsetRegisters[0]);

            asmjit::Label peel_loop_begin = cc->newLabel();
            asmjit::Label peel_loop_end = cc->newLabel();
            asmjit::Label reminder_loop_begin = cc->newLabel();
            asmjit::Label reminder_loop_end = cc->newLabel();
            asmjit::Label exit = cc->newLabel();

            err = cc->cmp(cnt, SIMD_STRIDE);
            err = cc->jl(peel_loop_end); // skip the peel loop if element count too small

            err = cc->bind(peel_loop_begin);
            {
                // SIMD loop
                asmjit::X86Ymm dst = cc->newYmmPs();
                eval_simd(reinterpret_exp, dst);

                cc->vmovaps(asmjit::x86::yword_ptr(offsetRegisters[0]), dst);

                // Advance loop
                //    Advance destination register
                cc->add(offsetRegisters[0], sizeof(SCALAR_T)*SIMD_STRIDE);
                //    Advance other registers
                for (int i = 1; i < argCount; i++)
                {
                    if (isScalar[i] == false)
                    {
                        cc->add(offsetRegisters[i], sizeof(SCALAR_T)*SIMD_STRIDE);
                    }
                }
            }
            err = cc->add(cnt, -(int)SIMD_STRIDE);
            err = cc->cmp(cnt, SIMD_STRIDE);
            err = cc->jge(peel_loop_begin);
            err = cc->bind(peel_loop_end);

            // Check if reminder present
            err = cc->test(cnt, cnt);
            err = cc->jz(exit);  // Exit if no reminder

                                // Scalar code to handle reminder
            err = cc->bind(reminder_loop_begin);
            {
                // scalar loop
            }
            err = cc->dec(cnt);
            err = cc->jnz(reminder_loop_begin);

            err = cc->bind(exit);
            cc->ret();
        }
        cc->endFunc(); // Close the evaluator function
        //err = cc->finalize();
        /*err = runtime.add(, &code);
        if (err) {
            assert(false);
        }*/

        // Build wrapper caller
        // Destroy the old compiler, and build a new one.
        //delete cc;

        //asmjit::CodeHolder wrapperCode;
        //wrapperCode.init(runtime.getCodeInfo());
        //cc = new asmjit::X86Compiler(&wrapperCode);

        asmjit::CCFunc* wrapper_func = cc->addFunc(asmjit::FuncSignature0<void>()); // add a trivial function

        asmjit::CCFuncCall* call = cc->call(evaluator->getLabel(), evaluator_signature);

        // Set actual arguments
//        asmjit::X86Gp t0 = cc->newInt32();
  //      cc->mov(t0, dst.LENGTH());
    //    call->setArg(0, t0);
        //call->setArg(1, asmjit::X86Mem(uint64_t(dst.elements)));

        for (int i = 1; i < argCount; i++) {
            //  call->setArg(i + 1, arguments[i]);
        }

        cc->endFunc();
        cc->finalize();

        wrapperFunc fun;
        //err = runtime.add(&fun, &wrapperCode);
        err = runtime.add(&fun, &code);
        if (err) {
            assert(false);
        }

        // Call the wrapper function
        fun();

        delete cc;
    }

    template<typename SCALAR_T> 
    void map_arguments(UME::VECTOR::FloatVector<SCALAR_T, VEC_LEN, SIMD_STRIDE> exp)
    {
        // We have to register every terminal to get proper function call parameters.
        // register a new pointer-type argument
        evaluator_signature.addArg(asmjit::TypeIdOf<SCALAR_T*>::kTypeId);
        // remember the terminal address
        arguments[argCount] = (uint64_t)exp.elements;
        isScalar[argCount] = false;
        // allocate an offset register
        offsetRegisters[argCount] = cc->newIntPtr();
        argCount++;
    }

    template<typename SCALAR_T, typename E1, typename E2>
    void map_arguments(UME::VECTOR::ArithmeticADDExpression<SCALAR_T, SIMD_STRIDE, E1, E2> exp)
    {
        // No need to mapp an ADD node. Map children in left-to-right order.
        map_arguments(exp._e1);
        map_arguments(exp._e2);
    }

    template<typename SCALAR_T>
    void eval_simd(UME::VECTOR::FloatVector<SCALAR_T, VEC_LEN, SIMD_STRIDE> & exp, asmjit::X86Ymm & dst) {
        // 1. Find the offset register from mapping
        int id = -1;
        for (int i = 0; i < argCount; i++) {
            if (arguments[i] == (uint64_t)exp.elements) {
                id = i;
                break;
            }
        }
        assert(id >= 0);

        // 2. Add a load operation to the operation list
        cc->vmovaps(dst, asmjit::x86::yword_ptr(offsetRegisters[id]));
    }

    template<typename SCALAR_T>
    void eval_scalar(UME::VECTOR::FloatVector<SCALAR_T, VEC_LEN, SIMD_STRIDE> exp, asmjit::X86Xmm & dst) {
        // 1. Find the offset register from mapping
        int id = -1;
        for (int i = 0; i < argCount; i++) {
            if (arguments[i] == exp.elements) {
                id = i;
                break;
            }
        }
        assert(id >= 0);

        // 2. Add a load operation to the operation list
        cc->movss(dst, asmjit::x86::ptr(offsetRegisters[id]));
    }


    // TODO: this has to be specialized!
    /*template<typename SCALAR_T, typename E1, typename E2>
    void eval_simd(UME::VECTOR::ArithmeticADDExpression<SCALAR_T, SIMD_STRIDE, E1, E2> exp) {
        assert(false);
    }*/

    template<typename SCALAR_T, typename E1, typename E2>
    void eval_simd(UME::VECTOR::ArithmeticADDExpression<SCALAR_T, SIMD_STRIDE, E1, E2> & exp, asmjit::X86Ymm & dst) {
        asmjit::X86Ymm t0 = cc->newYmmPs();
        asmjit::X86Ymm t1 = cc->newYmmPs();
        eval_simd(exp._e1, t0);
        eval_simd(exp._e2, t1);

        cc->vaddps(dst, t0, t1);
    }

};


template<typename FLOAT_T>
class UMEVectorAsmjitSingleTest : public AxpySingleTest<FLOAT_T>{
private:

   // typedef void(*Func)(int N, float alpha, float* x, float* y);
    //Func fn;

public:

    UMEVectorAsmjitSingleTest(int problem_size) : AxpySingleTest<FLOAT_T>(problem_size) {}

    ~UMEVectorAsmjitSingleTest() {}


    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        UME::VECTOR::Vector<FLOAT_T> x_vec(this->problem_size, this->x);
        UME::VECTOR::Vector<FLOAT_T> y_vec(this->problem_size, this->y);

        auto t0 =  x_vec + y_vec;
        AsmjitEvaluator<UME_DYNAMIC_LENGTH, UME_DEFAULT_SIMD_STRIDE> eval(y_vec, t0);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "AsmJIT(UME::VECTOR) single, (" +
            ScalarToString<FLOAT_T>::value() + ") " +
            std::to_string(this->problem_size);
        return retval;
    }
};

template<typename FLOAT_T>
class UMEVectorAsmjitChainedTest : public AxpyChainedTest<FLOAT_T> {
private:
    bool isCodeGenerated;

    // JIT structures
    asmjit::JitRuntime runtime;
    asmjit::CodeHolder code;

    typedef void(*Func)(
        int N,
        float* alpha,
        float* x0, float* x1,
        float* x2, float* x3,
        float* x4, float* x5,
        float* x6, float* x7,
        float* x8, float* x9,
        float* y);

    Func fn;
public:
    UMEVectorAsmjitChainedTest(int problem_size) : AxpyChainedTest<FLOAT_T>(problem_size)
    {
        code.init(runtime.getCodeInfo());

        if (!isCodeGenerated)
        {
            asmjit::Error err;

            asmjit::X86Compiler cc(&code);

            //cc.addFunc(asmjit::FuncSignature2<void, float*, float*>());
            cc.addFunc(asmjit::FuncSignatureT<void, int, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*>());

            ////////////////////////////////////////////
            // Declare registers
            asmjit::X86Gp cnt = cc.newInt32("cnt");
            asmjit::X86Gp alpha_off_reg = cc.newIntPtr("alphs_off_reg");
            asmjit::X86Gp x0_off_reg = cc.newIntPtr("x0_off_reg");
            asmjit::X86Gp x1_off_reg = cc.newIntPtr("x1_off_reg");
            asmjit::X86Gp x2_off_reg = cc.newIntPtr("x2_off_reg");
            asmjit::X86Gp x3_off_reg = cc.newIntPtr("x3_off_reg");
            asmjit::X86Gp x4_off_reg = cc.newIntPtr("x4_off_reg");
            asmjit::X86Gp x5_off_reg = cc.newIntPtr("x5_off_reg");
            asmjit::X86Gp x6_off_reg = cc.newIntPtr("x6_off_reg");
            asmjit::X86Gp x7_off_reg = cc.newIntPtr("x7_off_reg");
            asmjit::X86Gp x8_off_reg = cc.newIntPtr("x8_off_reg");
            asmjit::X86Gp x9_off_reg = cc.newIntPtr("x9_off_reg");
            asmjit::X86Gp y_off_reg = cc.newIntPtr("y_off_reg");
            asmjit::X86Ymm result = cc.newYmmPs("result");
            asmjit::X86Xmm alpha0 = cc.newXmmPs("alpha0");
            asmjit::X86Xmm alpha1 = cc.newXmmPs("alpha1");
            asmjit::X86Xmm alpha2 = cc.newXmmPs("alpha2");
            asmjit::X86Xmm alpha3 = cc.newXmmPs("alpha3");
            asmjit::X86Xmm alpha4 = cc.newXmmPs("alpha4");
            asmjit::X86Xmm alpha5 = cc.newXmmPs("alpha5");
            asmjit::X86Xmm alpha6 = cc.newXmmPs("alpha6");
            asmjit::X86Xmm alpha7 = cc.newXmmPs("alpha7");
            asmjit::X86Xmm alpha8 = cc.newXmmPs("alpha8");
            asmjit::X86Xmm alpha9 = cc.newXmmPs("alpha9");

            asmjit::X86Ymm alpha0_vec = cc.newYmmPs("alpha0_vec");
            asmjit::X86Ymm alpha1_vec = cc.newYmmPs("alpha1_vec");
            asmjit::X86Ymm alpha2_vec = cc.newYmmPs("alpha2_vec");
            asmjit::X86Ymm alpha3_vec = cc.newYmmPs("alpha3_vec");
            asmjit::X86Ymm alpha4_vec = cc.newYmmPs("alpha4_vec");
            asmjit::X86Ymm alpha5_vec = cc.newYmmPs("alpha5_vec");
            asmjit::X86Ymm alpha6_vec = cc.newYmmPs("alpha6_vec");
            asmjit::X86Ymm alpha7_vec = cc.newYmmPs("alpha7_vec");
            asmjit::X86Ymm alpha8_vec = cc.newYmmPs("alpha8_vec");
            asmjit::X86Ymm alpha9_vec = cc.newYmmPs("alpha9_vec");

            asmjit::X86Ymm t0 = cc.newYmmPs("t0");
            asmjit::X86Ymm t1 = cc.newYmmPs("t1");
            asmjit::X86Ymm t2 = cc.newYmmPs("t2");

            //asmjit::X86Mem alpha = cc.newFloatConst(asmjit::kConstScopeLocal, this->alpha);
            //asmjit::X86Mem N = cc.newInt32Const(asmjit::kConstScopeLocal, this->problem_size);

            // Define input mapping
            cc.setArg(0, cnt);
            cc.setArg(1, alpha_off_reg);
            cc.setArg(2, x0_off_reg);
            cc.setArg(3, x1_off_reg);
            cc.setArg(4, x2_off_reg);
            cc.setArg(5, x3_off_reg);
            cc.setArg(6, x4_off_reg);
            cc.setArg(7, x5_off_reg);
            cc.setArg(8, x6_off_reg);
            cc.setArg(9, x7_off_reg);
            cc.setArg(10, x8_off_reg);
            cc.setArg(11, x9_off_reg);
            cc.setArg(12, y_off_reg);

            // Code generation

            asmjit::Label peel_loop_begin = cc.newLabel();
            asmjit::Label peel_loop_end = cc.newLabel();
            asmjit::Label reminder_loop_begin = cc.newLabel();
            asmjit::Label reminder_loop_end = cc.newLabel();
            asmjit::Label exit = cc.newLabel();

            err = cc.cmp(cnt, 8);
            err = cc.jl(peel_loop_end); // skip the peel loop if element count too small

            err = cc.movss(alpha0, asmjit::x86::ptr(alpha_off_reg, 0));
            err = cc.movss(alpha1, asmjit::x86::ptr(alpha_off_reg, 4));
            err = cc.movss(alpha2, asmjit::x86::ptr(alpha_off_reg, 8));
            err = cc.movss(alpha3, asmjit::x86::ptr(alpha_off_reg, 12));
            err = cc.movss(alpha4, asmjit::x86::ptr(alpha_off_reg, 16));
            err = cc.movss(alpha5, asmjit::x86::ptr(alpha_off_reg, 20));
            err = cc.movss(alpha6, asmjit::x86::ptr(alpha_off_reg, 24));
            err = cc.movss(alpha7, asmjit::x86::ptr(alpha_off_reg, 28));
            err = cc.movss(alpha8, asmjit::x86::ptr(alpha_off_reg, 32));
            err = cc.movss(alpha9, asmjit::x86::ptr(alpha_off_reg, 36));

            err = cc.vshufps(alpha0_vec, alpha0.ymm(), alpha0.ymm(), 0);
            err = cc.vshufps(alpha1_vec, alpha1.ymm(), alpha1.ymm(), 0);
            err = cc.vshufps(alpha2_vec, alpha2.ymm(), alpha2.ymm(), 0);
            err = cc.vshufps(alpha3_vec, alpha3.ymm(), alpha3.ymm(), 0);
            err = cc.vshufps(alpha4_vec, alpha4.ymm(), alpha4.ymm(), 0);
            err = cc.vshufps(alpha5_vec, alpha5.ymm(), alpha5.ymm(), 0);
            err = cc.vshufps(alpha6_vec, alpha6.ymm(), alpha6.ymm(), 0);
            err = cc.vshufps(alpha7_vec, alpha7.ymm(), alpha7.ymm(), 0);
            err = cc.vshufps(alpha8_vec, alpha8.ymm(), alpha8.ymm(), 0);
            err = cc.vshufps(alpha9_vec, alpha9.ymm(), alpha9.ymm(), 0);

            err = cc.bind(peel_loop_begin);
            {
                err = cc.vmulps(t0, alpha0_vec, asmjit::x86::ptr(x0_off_reg));
                err = cc.vmulps(t1, alpha1_vec, asmjit::x86::ptr(x1_off_reg));
                err = cc.vaddps(t2, t0, t1);

                err = cc.vmulps(t0, alpha2_vec, asmjit::x86::ptr(x2_off_reg));
                err = cc.vaddps(t1, t2, t0);

                err = cc.vmulps(t0, alpha3_vec, asmjit::x86::ptr(x3_off_reg));
                err = cc.vaddps(t2, t1, t0);

                err = cc.vmulps(t0, alpha4_vec, asmjit::x86::ptr(x4_off_reg));
                err = cc.vaddps(t1, t2, t0);

                err = cc.vmulps(t0, alpha5_vec, asmjit::x86::ptr(x5_off_reg));
                err = cc.vaddps(t2, t1, t0);

                err = cc.vmulps(t0, alpha6_vec, asmjit::x86::ptr(x6_off_reg));
                err = cc.vaddps(t1, t2, t0);

                err = cc.vmulps(t0, alpha7_vec, asmjit::x86::ptr(x7_off_reg));
                err = cc.vaddps(t2, t1, t0);

                err = cc.vmulps(t0, alpha8_vec, asmjit::x86::ptr(x8_off_reg));
                err = cc.vaddps(t1, t2, t0);

                err = cc.vmulps(t0, alpha9_vec, asmjit::x86::ptr(x9_off_reg));
                err = cc.vaddps(t2, t1, t0);

                err = cc.vaddps(result, t2, asmjit::x86::ptr(y_off_reg));
                err = cc.vmovaps(asmjit::x86::yword_ptr(y_off_reg), result.ymm());

                err = cc.add(x0_off_reg, 32);
                err = cc.add(x1_off_reg, 32);
                err = cc.add(x2_off_reg, 32);
                err = cc.add(x3_off_reg, 32);
                err = cc.add(x4_off_reg, 32);
                err = cc.add(x5_off_reg, 32);
                err = cc.add(x6_off_reg, 32);
                err = cc.add(x7_off_reg, 32);
                err = cc.add(x8_off_reg, 32);
                err = cc.add(x9_off_reg, 32);

                err = cc.add(y_off_reg, 32);

                err = cc.add(cnt, -8);
                err = cc.cmp(cnt, 8);
            }
            err = cc.jge(peel_loop_begin);
            err = cc.bind(peel_loop_end);

            // Check if reminder present
           /* err = cc.test(cnt, cnt);
            err = cc.jz(exit);  // Exit if no reminder

                                // Scalar code to handle reminder
            err = cc.bind(reminder_loop_begin);
            {
                /*
                err = cc.movss(result.xmm(), alpha.xmm());
                err = cc.mulss(result.xmm(), asmjit::x86::ptr(x_off_reg)); // multiply 'alpha' and x[i]
                err = cc.addss(result.xmm(), asmjit::x86::ptr(y_off_reg)); // add 'x*alpha' and 'y[i]'
                err = cc.movss(asmjit::x86::ptr(y_off_reg), result.xmm());

                err = cc.add(x_off_reg, 4);                          // Increment 'arr' pointer.
                err = cc.add(y_off_reg, 4);                          // Increment 'arr' pointer.
                
            }
            err = cc.dec(cnt);
            err = cc.jnz(reminder_loop_begin);
            err = cc.bind(reminder_loop_end);*/

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

    UME_NEVER_INLINE virtual void benchmarked_code()
    {
        fn(
            this->problem_size,
            this->alpha,
            this->x0, this->x1,
            this->x2, this->x3,
            this->x4, this->x5,
            this->x6, this->x7,
            this->x8, this->x9,
            this->y);
    }

    UME_NEVER_INLINE virtual std::string get_test_identifier()
    {
        std::string retval = "";
        retval += "AsmJIT(UME::VECTOR) chained, (" +
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