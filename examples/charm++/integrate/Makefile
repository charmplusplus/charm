CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm: pgm.o
	$(CHARMC) pgm.o  -o pgm -language charm++

pgm.o : pgm.C pgm.h pgm.ci
	$(CHARMC) -c pgm.ci pgm.C

clean:
	rm -f pgm *.o conv-host *.decl.h *.def.h *~ charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk

test: pgm
	./charmrun ./pgm +p4 100 5 $(TESTOPTS)

bgtest: pgm
	./charmrun ./pgm +p4 100 5 +x2 +y2 +z1 $(TESTOPTS)
