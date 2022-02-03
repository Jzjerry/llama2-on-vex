#pragma once

#define BASE 0x10000000
#define PUTC BASE
#define PUT_HEX (BASE + 0x8)
#define CLINT_BASE (BASE + 0x10000)
#define CLINT_CMP (CLINT_BASE + 0x4000)
#define CLINT_TIME (CLINT_BASE + 0x0BFF8)
#define MACHINE_EXTERNAL_INTERRUPT_CTRL (BASE+0x10)
#define SUPERVISOR_EXTERNAL_INTERRUPT_CTRL (BASE + 0x18)
#define GETC (BASE + 0x40)
#define INCR_COUNTER (BASE + 0x70)


#define MM_FAULT_ADDRESS 0x00001230
#define IO_FAULT_ADDRESS 0x1FFFFFF0


#define ICACHE_REFILL  1
#define DCACHE_REFILL  2
#define DCACHE_WRITEBACK  3
#define BRANCH_MISS  4