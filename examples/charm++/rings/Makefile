CHARMC=../../../bin/charmc $(OPTS)

pgm:  rings.o
	$(CHARMC) rings.o  -o pgm -language charm++

rings.o : rings.C rings.h rings.ci
	$(CHARMC) -c rings.ci rings.C

test: pgm
	./charmrun +p4 pgm

clean:
	rm -f conv-host pgm *.o *.decl.h *.def.h *~ charmrun
