PROJ_NAME=llama2

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

all: bin/model.o bin/tokenizer.o

bin/model.o: bin/model.bin
	riscv64-unknown-elf-ld -m elf32lriscv -r -b binary -o bin/model.o bin/model.bin

bin/tokenizer.o: bin/tokenizer.bin
	riscv64-unknown-elf-ld -m elf32lriscv -r -b binary -o bin/tokenizer.o bin/tokenizer.bin

clean_bin:
	rm bin/*.o

include app.mk