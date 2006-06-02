CHARMC=../../../bin/charmc $(OPTS)

pgm:  onesided.o
	$(CHARMC) onesided.o  -o onesidedtest -language charm++

onesided.o : onesided.C onesided.h onesided.ci
	$(CHARMC) -c onesided.ci 
	$(CHARMC) -c onesided.C

test: pgm
	./charmrun onesidedtest 0 +p2
	./charmrun onesidedtest 1 +p2

clean:
	rm -f conv-host onesidedtest *.def.h *.decl.h *.o *~ charmrun
