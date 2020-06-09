-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm: single.o
	$(CHARMC) single.o  -o pgm -language charm++

single.o : single.C single.h single.ci
	$(CHARMC) -c single.ci single.C

test: pgm
	$(call run, ++local +p4 ./pgm )

clean:
	rm -f conv-host pgm *.def.h *.decl.h *.o *~ charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk
