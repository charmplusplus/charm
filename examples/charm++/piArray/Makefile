CHARMC=../../../bin/charmc $(OPTS)

test: pgm
	./charmrun +p4 pgm 100000 100

bgtest: pgm
	./charmrun +p4 pgm 100000 100 +x2 +y2 +z1

pgm: pgm.o
	$(CHARMC) pgm.o  -o pgm -language charm++

pgm.o : pgm.C pgm.h pgm.decl.h
	$(CHARMC) -c pgm.C

pgm.decl.h: pgm.ci
	$(CHARMC) pgm.ci

clean:
	rm -f pgm *.o conv-host *.decl.h *.def.h *~ charmrun
