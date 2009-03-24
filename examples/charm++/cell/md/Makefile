CHARM_BASE_DIR = ../../../..
CHARM_BIN_DIR = $(CHARM_BASE_DIR)/bin
CHARM_INC_DIR = $(CHARM_BASE_DIR)/include
CHARM_LIB_DIR = $(CHARM_BASE_DIR)/lib
CHARMC = $(CHARM_BIN_DIR)/charmc $(OPTS)

PGM = md

default: all
all: $(PGM) $(PGM)_proj

OBJS = main.o patch.o selfCompute.o pairCompute.o
SPE_LIBS = -spu-lib m
ACCEL_LIBS = $(SPE_LIBS)


################################################################################
## Linker Commands

# Standard Build (no projections)
$(PGM): $(OBJS)
	$(CHARMC) -language charm++ -o $(PGM) $(OBJS) $(ACCEL_LIBS)

# Projections Build
$(PGM)_proj: $(OBJS)
	$(CHARMC) -language charm++ -o $(PGM)_proj $(OBJS) $(ACCEL_LIBS) -tracemode summary -tracemode projections


################################################################################
## Debug: SPE Timing (Cell specific)

speCode.s.timing : $(PGM)
	spu-gcc $(OPTS) -I$(CHARM_INC_DIR) -S -c main__funcLookup__.genSPECode.c -o speCode.s -DCMK_CELL_SPE=1
	/opt/cell/sdk/usr/bin/spu_timing speCode.s


################################################################################
## Chare Classes


### Main
main.decl.h main.def.h: main.ci
	$(CHARMC) main.ci

main.o: main.h main.C main.decl.h patch.decl.h selfCompute.decl.h pairCompute.decl.h main.def.h
	$(CHARMC) -c main.C


### Patch
patch.decl.h patch.def.h: patch.ci
	$(CHARMC) patch.ci

patch.o: patch.h main.h patch.C patch.decl.h main.decl.h pairCompute.decl.h patch.def.h
	$(CHARMC) -c patch.C


### SelfCompute
selfCompute.decl.h selfCompute.def.h: selfCompute.ci
	$(CHARMC) selfCompute.ci

selfCompute.o: selfCompute.h main.h selfCompute.C selfCompute.decl.h patch.decl.h selfCompute.def.h
	$(CHARMC) -c selfCompute.C


### PairCompute
pairCompute.decl.h pairCompute.def.h: pairCompute.ci
	$(CHARMC) pairCompute.ci

pairCompute.o: pairCompute.h main.h pairCompute.C pairCompute.decl.h patch.decl.h pairCompute.def.h
	$(CHARMC) -c pairCompute.C


################################################################################
## Binary File, Object Files, etc. Cleanup

clean:
	rm -f *.decl.h *.def.h conv-host *.o $(PGM) charmrun *genSPECode* speCode.s speCode.s.timing
