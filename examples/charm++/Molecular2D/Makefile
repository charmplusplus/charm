CHARMC	=../../../bin/charmc $(OPTS)
OPTS	= -O3

all: mol2d

mol2d: Patch.o Compute.o
	$(CHARMC) -language charm++ -o mol2d Patch.o Compute.o

Patch.o: Patch.C Patch.h Patch.decl.h common.h
	$(CHARMC) -o Patch.o Patch.C

Patch.decl.h: Patch.ci
	$(CHARMC) Patch.ci

Compute.o: Compute.C Compute.h Patch.decl.h common.h
	$(CHARMC) -o Compute.o Compute.C

clean:
	rm -f *.decl.h *.def.h *.o mol2d charmrun
