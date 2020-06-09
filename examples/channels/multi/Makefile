-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm: multi.o
	$(CHARMC) multi.o  -o pgm -language charm++

multi.o : multi.C multi.h multi.ci
	$(CHARMC) -c multi.ci multi.C

test: pgm
	$(call run, ++local +p4 ./pgm )

clean:
	rm -f conv-host pgm *.def.h *.decl.h *.o *~ charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk
