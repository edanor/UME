#pragma once

#include "../../UME.h"

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

    static const int MAX_ARG_COUNT = 128;

    int argCount;

    // Argument 1 is the destination, but since argument 0 is number of elements, we 
    // dont store it here. 'arguments[0]' is then destination pointer.
    uint64_t arguments[MAX_ARG_COUNT];
    // Offset registers for each pointer argument
    asmjit::X86Gp offsetRegisters[MAX_ARG_COUNT];
    // value registers for each scalar value

    typedef void(*evaluatorFunc)(void);

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

        asmjit::X86Gp cnt = cc->newInt32("cnt");
        offsetRegisters[0] = cc->newIntPtr("Dst");

        // Visit all nodes and figure out the function signature.
        argCount = 1;
        map_arguments(reinterpret_exp);

        // Now when all arguments are already mapped, we can initialize parameters
        asmjit::CCFunc* evaluator = cc->addFunc(asmjit::FuncSignature0<void>());

        // Initialize iteration count
        asmjit::X86Mem t0 = cc->newInt64Const(asmjit::kConstScopeLocal, dst.LENGTH());
        cc->mov(cnt, t0);

        // Initialize destination offset register
        asmjit::X86Mem t1 = cc->newInt64Const(asmjit::kConstScopeLocal, (uint64_t)dst.elements);
        cc->mov(offsetRegisters[0], t1);

        // Initialize remaining offset registers
        for (int i = 1; i < argCount; i++) {
            asmjit::X86Mem tX = cc->newInt64Const(asmjit::kConstScopeLocal, arguments[i]);
            err = cc->mov(offsetRegisters[i], tX);
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

                err = cc->vmovaps(asmjit::x86::yword_ptr(offsetRegisters[0]), dst);


                // Advance loop
                //    Advance destination register
                err = cc->add(offsetRegisters[0], sizeof(SCALAR_T)*SIMD_STRIDE);
                //    Advance other registers
                for (int i = 1; i < argCount; i++)
                {
                    cc->add(offsetRegisters[i], sizeof(SCALAR_T)*SIMD_STRIDE);
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
                asmjit::X86Xmm dst = cc->newXmmPs();
                eval_scalar(reinterpret_exp, dst);

                err = cc->movss(asmjit::x86::ptr(offsetRegisters[0]), dst);

                // Advance loop
                //    Advance destination register
                err = cc->add(offsetRegisters[0], sizeof(SCALAR_T));
                //    Advance other registers
                for (int i = 1; i < argCount; i++)
                {
                    cc->add(offsetRegisters[i], sizeof(SCALAR_T));
                }
            }
            err = cc->dec(cnt);
            err = cc->jnz(reminder_loop_begin);

            err = cc->bind(exit);
            cc->ret();
        }
        cc->endFunc(); // Close the evaluator function

        cc->finalize();

        evaluatorFunc eval;
        //err = runtime.add(&fun, &wrapperCode);
        err = runtime.add(&eval, &code);
        if (err) {
            assert(false);
        }

        // Call the wrapper function
        eval();

        delete cc;
    }

    template<typename SCALAR_T>
    UME_FORCE_INLINE void map_arguments(UME::VECTOR::Scalar<SCALAR_T, SIMD_STRIDE> exp)
    {
        // Do nothing. The value can be bound at the moment of scalar evaluation.
    }

    template<typename SCALAR_T>
    UME_FORCE_INLINE void map_arguments(UME::VECTOR::FloatVector<SCALAR_T, VEC_LEN, SIMD_STRIDE> exp)
    {
        // We have to register every terminal to get proper initial values.
        // remember the terminal address
        arguments[argCount] = (uint64_t)exp.elements;
        // allocate an offset register
        offsetRegisters[argCount] = cc->newIntPtr();
        argCount++;
    }

    template<typename SCALAR_T, typename E1, typename E2>
    UME_FORCE_INLINE void map_arguments(UME::VECTOR::ArithmeticADDExpression<SCALAR_T, SIMD_STRIDE, E1, E2> exp)
    {
        // No need to mapp an ADD node. Map children in left-to-right order.
        map_arguments(exp._e1);
        map_arguments(exp._e2);
    }

    template<typename SCALAR_T, typename E1, typename E2>
    UME_FORCE_INLINE void map_arguments(UME::VECTOR::ArithmeticMULExpression<SCALAR_T, SIMD_STRIDE, E1, E2> exp)
    {
        map_arguments(exp._e1);
        map_arguments(exp._e2);
    }

    // TODO: we need a dispatch to make differentiation between SIMD strides and register mappings
    // TODO: scalars can be initialized before the main loop, so that excess broadcasts don't happen
    template<typename SCALAR_T>
    UME_FORCE_INLINE void eval_simd(UME::VECTOR::Scalar<SCALAR_T, SIMD_STRIDE> & exp, asmjit::X86Ymm & dst) {
        //asmjit::X86Mem t0 = cc->newFloatConst(asmjit::kConstScopeLocal, exp._e1);
        asmjit::X86Mem t1 = cc->newYmmConst(asmjit::kConstScopeGlobal, asmjit::Data256().fromF32(exp._e1));
        auto err = cc->vmovaps(dst, t1);
        assert(err == 0);
    }

    template<typename SCALAR_T>
    UME_FORCE_INLINE void eval_scalar(UME::VECTOR::Scalar<SCALAR_T, SIMD_STRIDE> & exp, asmjit::X86Xmm & dst) {
        //asmjit::X86Mem t0 = cc->newFloatConst(asmjit::kConstScopeLocal, exp._e1);
        asmjit::X86Mem t1 = cc->newXmmConst(asmjit::kConstScopeGlobal, asmjit::Data128().fromF32(exp._e1));
        auto err = cc->movss(dst, t1);
        assert(err == 0);
    }

    template<typename SCALAR_T>
    UME_FORCE_INLINE void eval_simd(UME::VECTOR::FloatVector<SCALAR_T, VEC_LEN, SIMD_STRIDE> & exp, asmjit::X86Ymm & dst) {
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
        auto err = cc->vmovaps(dst, asmjit::x86::yword_ptr(offsetRegisters[id]));
        assert(err == 0);
    }

    template<typename SCALAR_T>
    UME_FORCE_INLINE void eval_scalar(UME::VECTOR::FloatVector<SCALAR_T, VEC_LEN, SIMD_STRIDE> exp, asmjit::X86Xmm & dst) {
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
        auto err = cc->movss(dst, asmjit::x86::ptr(offsetRegisters[id]));
        assert(err == 0);
    }


    // TODO: this has to be specialized!
    /*template<typename SCALAR_T, typename E1, typename E2>
    void eval_simd(UME::VECTOR::ArithmeticADDExpression<SCALAR_T, SIMD_STRIDE, E1, E2> exp) {
    assert(false);
    }*/

    template<typename SCALAR_T, typename E1, typename E2>
    UME_FORCE_INLINE void eval_simd(UME::VECTOR::ArithmeticADDExpression<SCALAR_T, SIMD_STRIDE, E1, E2> & exp, asmjit::X86Ymm & dst) {
        asmjit::X86Ymm t0 = cc->newYmmPs();
        asmjit::X86Ymm t1 = cc->newYmmPs();
        eval_simd(exp._e1, t0);
        eval_simd(exp._e2, t1);

        auto err = cc->vaddps(dst, t0, t1);
        assert(err == 0);
    }

    template<typename SCALAR_T, typename E1, typename E2>
    UME_FORCE_INLINE void eval_scalar(UME::VECTOR::ArithmeticADDExpression<SCALAR_T, SIMD_STRIDE, E1, E2> & exp, asmjit::X86Xmm & dst) {
        asmjit::X86Xmm t0 = cc->newXmmPs();
        asmjit::X86Xmm t1 = cc->newXmmPs();
        eval_scalar(exp._e1, t0);
        eval_scalar(exp._e2, t1);

        auto err = cc->vaddps(dst, t0, t1);
        assert(err == 0);
    }

    template<typename SCALAR_T, typename E1, typename E2>
    UME_FORCE_INLINE void eval_simd(UME::VECTOR::ArithmeticMULExpression<SCALAR_T, SIMD_STRIDE, E1, E2> & exp, asmjit::X86Ymm & dst) {
        asmjit::X86Ymm t0 = cc->newYmmPs();
        asmjit::X86Ymm t1 = cc->newYmmPs();
        eval_simd(exp._e1, t0);
        eval_simd(exp._e2, t1);

        auto err = cc->vmulps(dst, t0, t1);
        assert(err == 0);
    }


    template<typename SCALAR_T, typename E1, typename E2>
    UME_FORCE_INLINE void eval_scalar(UME::VECTOR::ArithmeticMULExpression<SCALAR_T, SIMD_STRIDE, E1, E2> & exp, asmjit::X86Xmm & dst) {
        asmjit::X86Xmm t0 = cc->newXmmPs();
        asmjit::X86Xmm t1 = cc->newXmmPs();
        eval_scalar(exp._e1, t0);
        eval_scalar(exp._e2, t1);

        auto err = cc->vmulps(dst, t0, t1);
        assert(err == 0);
    }

};
