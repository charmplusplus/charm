CHARMC=../../../bin/charmc $(OPTS)

pgm:  ring.o
	$(CHARMC) ring.o  -o pgm -language charm++

ring.o : ring.C ring.h ring.ci
	$(CHARMC) -c ring.ci ring.C

test: pgm
	./charmrun +p4 pgm

clean:
	rm -f conv-host pgm *.def.h *.decl.h *.o *~ charmrun
