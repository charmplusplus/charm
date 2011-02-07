CHARMC=../../../bin/charmc $(OPTS)

all: pgm

pgm: pgm.o cst_nl.o
	$(CHARMC) -o pgm pgm.o cst_nl.o -language ParFUM -module netfem 

pgm.o: pgm.h pgm.C
	$(CHARMC) -c pgm.C

cst_nl.o: pgm.h cst_nl.C
	$(CHARMC) -c cst_nl.C


test: pgm
	./charmrun ./pgm +vp4 +p2 $(TESTOPTS)

bgtest: pgm
	./charmrun ./pgm +vp4 +p2 +x2 +y2 +z1 $(TESTOPTS)

clean:
	rm -f pgm fpgm *.o conv-host charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk
