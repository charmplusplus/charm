-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm: stress.o
	$(CHARMC) stress.o  -o pgm -language charm++

stress.o : stress.C stress.h stress.ci
	$(CHARMC) -c stress.ci stress.C

test: pgm
	$(call run, ++local +p4 ./pgm )

clean:
	rm -f conv-host pgm *.def.h *.decl.h *.o *~ charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk
