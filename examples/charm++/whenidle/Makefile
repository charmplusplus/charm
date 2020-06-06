-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm:  idlework.o
	$(CHARMC) idlework.o  -o pgm -language charm++

idlework.o : idlework.C idlework.h idlework.ci
	$(CHARMC) -c idlework.ci idlework.C

test: pgm
	$(call run, +p4 ./pgm )

clean:
	rm -f conv-host pgm *.def.h *.decl.h *.o *~ charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk
