PROJ_NAME=bitnet

DEBUG?=no
BENCH=yes
STANDALONE = ..
MARCH = rv32imc

USE_SIMD?=8
BITNET_QUANT?=2

CFLAGS += -fno-common -fno-inline
CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
CFLAGS += -DTIME -DCORE_HZ=12000000ll
CFLAGS += -DUSE_SIMD=$(USE_SIMD) -DBITNET_QUANT=$(BITNET_QUANT)
CFLAGS += -I./

SRCS = 	$(wildcard *.c) \
		$(wildcard *.S) \
        ${STANDALONE}/common/start.S


BIN_OBJS = bin/model.o

all: bin/model.o

bin/model.o: bin/model.bin
	riscv64-unknown-elf-ld -m elf32lriscv -r -b binary -o bin/model.o bin/model.bin

clean_bin:
	rm bin/*.o

include app.mk