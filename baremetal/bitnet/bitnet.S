#include "custom_asm.h"

.global __bitnetadd4
__bitnetadd4:
    opcode_R(CUSTOM0, 0x1, 0x00, a0, a0, a1) // BNADD4 a0 = a(a0) * w(a1)
    ret

.global __bitnetadd8
__bitnetadd8:
    opcode_R(CUSTOM0, 0x2, 0x00, x0, x0, a2) // BNSTORE a2 -> buffer
    opcode_R(CUSTOM0, 0x1, 0x00, a0, a0, a1) // BNADD8 a0 = a(a0) * w(a1)
    ret