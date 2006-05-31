CHARM_BASE_DIR = ../../../..
CHARM_BIN_DIR = $(CHARM_BASE_DIR)/bin
CHARMC = $(CHARM_BIN_DIR)/charmc $(OPTS)

GENERAL_DEPENDS = Makefile

# NOTE : Mainfile.cell will set some variables including : CELL_SDK_DIR, SPU_CXX, PPU_EMBEDSPU, etc.
include $(CHARM_BASE_DIR)/tmp/Makefile.cell


default : all
all : jacobi


#///////////////////////////////////////////////////////////////////////////////
#// General Charm++ Stuff

# Rule to combine all the object files
jacobi : main.o jacobi.o jacobi_shared.o
	$(CHARMC) -language charm++ -o jacobi main.o jacobi.o jacobi_shared.o

# Rule to compile the Main class
main.o : main.C main.h jacobi_shared.h main.decl.h jacobi.decl.h $(GENERAL_DEPENDS)
	$(CHARMC) -c -o main.o main.C

# Rule to compile the Jacobi class
jacobi.o : jacobi.C jacobi.h jacobi_shared.h jacobi.decl.h main.decl.h $(GENERAL_DEPENDS)
#	$(CHARMC) -c -o jacobi.o jacobi.C jacobi_shared.cpp
	$(CHARMC) -c -o jacobi.o jacobi.C

# General rule for creating decl and def files
%.decl.h : %.ci $(GENERAL_DEPENDS)
	$(CHARMC) $*.ci


#///////////////////////////////////////////////////////////////////////////////
#// Cell Specific Stuff

# Rule to create the embeded spu runtime that includes the user's shared code
jacobi_shared.o : jacobi_shared.cpp jacobi_shared.h $(GENERAL_DEPENDS)
#	$(SPU_CXX) -I$(CHARM_BASE_DIR)/include -I$(CELL_SDK_DIR)/sysroot/usr/spu/include -L$(CHARM_BASE_DIR)/lib -o jacobi_shared jacobi_shared.cpp -lcellspu
#	$(SPU_CXX) -I$(CHARM_BASE_DIR)/include -I$(CELL_SDK_DIR)/sysroot/usr/spu/include -L$(CHARM_BASE_DIR)/lib -L$(CELL_SDK_DIR)/sysroot/usr/spu/lib -nodefaultlibs -o jacobi_shared jacobi_shared.cpp -lcellspu -lc -lsim
	cp jacobi_shared.cpp jacobi_shared.c
#	spu-gcc $(OPTS) -I$(CHARM_BASE_DIR)/include -I$(CELL_SDK_DIR)/sysroot/usr/spu/include -o jacobi_shared jacobi_shared.c -lcellspu -lsim
#	spu-gcc $(OPTS) -c -I$(CHARM_BASE_DIR)/include -I$(CELL_SDK_DIR)/sysroot/usr/spu/include -L$(CHARM_BASE_DIR)/lib -L$(CELL_SDK_DIR)/sysroot/usr/spu/lib -o jacobi_shared_tmp.o jacobi_shared.c
#	spu-gcc -L$(CHARM_BASE_DIR)/lib -L$(CELL_SDK_DIR)/sysroot/usr/spu/lib -o jacobi_shared jacobi_shared_tmp.o -lcellspu -lsim
	$(SPU_CC) -I$(CHARM_BASE_DIR)/include -I$(CELL_SDK_DIR)/sysroot/usr/spu/include -L$(CHARM_BASE_DIR)/lib -L$(CELL_SDK_DIR)/sysroot/usr/spu/lib -o jacobi_shared jacobi_shared.c -lcellspu -lsim -lc
	$(PPU_EMBEDSPU) spert_main jacobi_shared jacobi_shared.o


#///////////////////////////////////////////////////////////////////////////////
#// Clean-up Stuff

clean :
	$(RM) *.decl.h *.def.h *.o jacobi_shared jacobi charmrun
