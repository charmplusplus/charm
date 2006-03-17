CELL_SDK_DIR = /home/strife/cellsim

CHARM_BASE_DIR = ../../../..
CHARM_BIN_DIR = $(CHARM_BASE_DIR)/bin
CHARMC = $(CHARM_BIN_DIR)/charmc $(OPTS)

SPU_CC = spu-gcc $(OPTS)
SPU_CXX = spu-g++ $(OPTS)
SPU_AR = spu-ar
PPU_EMBEDSPU = ppu32-embedspu

OBJS = hello.o

all: hello

hello: $(OBJS) hello_shared.o
	$(CHARMC) -language charm++ -o hello hello_shared.o $(OBJS)

hello_shared.o: hello_shared.cpp hello_shared.h
	$(SPU_CXX) -I$(CELL_SDK_DIR)/sysroot/usr/spu/include -L$(CHARM_BASE_DIR)/lib -o hello_shared hello_shared.cpp -lcellspu
	$(PPU_EMBEDSPU) spert_main hello_shared hello_shared.o

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun hello_shared

