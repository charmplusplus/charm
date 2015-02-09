-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm:  rings.o
	$(CHARMC) rings.o  -o pgm -language charm++

rings.o : rings.C rings.h rings.ci
	$(CHARMC) -c rings.ci rings.C

test: pgm
	$(call run, +p4 ./pgm )

bgtest: pgm
	$(call run, +p4 ./pgm +x2 +y2 +z2 )

clean:
	rm -f conv-host pgm *.o *.decl.h *.def.h *~ charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk
