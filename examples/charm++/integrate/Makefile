CHARMC=../../../bin/charmc $(OPTS)

pgm: pgm.o
	$(CHARMC) pgm.o  -o pgm -language charm++

pgm.o : pgm.C pgm.h pgm.ci
	$(CHARMC) -c pgm.ci pgm.C

clean:
	rm -f pgm *.o conv-host *.decl.h *.def.h *~ charmrun
