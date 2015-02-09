-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm:  ring.o
	$(CHARMC) ring.o  -o pgm -language charm++

ring.o : ring.C ring.h ring.ci
	$(CHARMC) -c ring.ci ring.C

test: pgm
	$(call run, +p4 ./pgm )

bgtest: pgm
	$(call run, +p4 ./pgm +x2 +y2 +z2 )

clean:
	rm -f conv-host pgm *.def.h *.decl.h *.o *~ charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk
