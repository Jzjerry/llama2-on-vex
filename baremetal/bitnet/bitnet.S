#include "custom_asm.h"

// For BitNet SIMD Below 8 elements, the arguments are the data themselves
.global __bitnetadd4
__bitnetadd4:
    opcode_R(CUSTOM0, 0x1, 0x00, a0, a0, a1) // BNADD4 a0 = a(a0) * w(a1)
    ret

.global __bitnetadd8
__bitnetadd8:
    opcode_R(CUSTOM0, 0x2, 0x00, x0, x0, a2) // BNSTORE a2 -> buffer
    opcode_R(CUSTOM0, 0x1, 0x00, a0, a0, a1) // BNADD8 a0 = a(a0) * w(a1)
    ret

// For BitNet SIMD Above 8 elements, the arguments are the data pointers
.global __bitnetadd16
__bitnetadd16:
    opcode_R(CUSTOM0, 0x2, 0x00, x0, x0, a1) // BNSTORE a1 -> buffer
    lw t0, 0(a0)
    lw t1, 4(a0)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    lw t0, 8(a0)
    lw t1, 12(a0)
    opcode_R(CUSTOM0, 0x1, 0x00, t1, t0, t1) // BNADD8 a0
    add a0, t2, t1
    ret

.global __bitnetadd32
__bitnetadd32:
    opcode_R(CUSTOM0, 0x2, 0x00, x0, a2, a1) // BNSTORE a2 ## a1 -> buffer
    li a3, 0 // offset pointer
    li a5, 0 // result buffer
    li t3, 32
loop4:
    add a4, a3, a0
    lw t0, 0(a4)
    lw t1, 4(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    addi a3, a3, 8
    add a5, a5, t2
    bne a3, t3, loop4
    add a0, a5, x0
    ret

// Available for 1-bit BitNet Only
.global __bitnetadd64
__bitnetadd64:
    opcode_R(CUSTOM0, 0x2, 0x00, x0, a2, a1) // BNSTORE a2 ## a1 -> buffer
    li a3, 0 // offset pointer
    li a5, 0 // result buffer
    li t3, 64
loop8:
    add a4, a3, a0
    lw t0, 0(a4)
    lw t1, 4(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    addi a3, a3, 8
    add a5, a5, t2
    bne a3, t3, loop8
    add a0, a5, x0
    ret