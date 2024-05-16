PROJ_NAME=bitnet

DEBUG?=no
BENCH=yes
STANDALONE = ..
MARCH = rv32imc

CFLAGS += -fno-common -fno-inline
CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
CFLAGS += -DTIME -DCORE_HZ=12000000ll
CFLAGS += -I./

SRCS = 	$(wildcard *.c) \
		$(wildcard *.S) \
        ${STANDALONE}/common/start.S


BIN_OBJS = $(wildcard bin/*.o)

all: bin/model.o

bin/model.o: bin/model.bin
	riscv64-unknown-elf-ld -m elf32lriscv -r -b binary -o bin/model.o bin/model.bin

clean_bin:
	rm bin/*.o

include app.mk