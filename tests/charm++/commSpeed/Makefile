CHARMC=../../../bin/charmc $(OPTS)

all: commSpeed

commSpeed: pgm.o Worker.o
	$(CHARMC) -language charm++ -o commSpeed pgm.o Worker.o

Pgm.decl.h: pgm.ci 
	$(CHARMC)  pgm.ci

pgm.o: pgm.C Pgm.decl.h Worker.o
	$(CHARMC) -c pgm.C

Worker.o: Worker.C Worker.decl.h 
	$(CHARMC) -language charm++ Worker.C

Worker.decl.h: Worker.ci
	$(CHARMC)  Worker.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o commSpeed charmrun *~

test: all
	./charmrun commSpeed +p1 100 100 $(TESTOPTS)
	./charmrun commSpeed +p2 100 100 $(TESTOPTS)
	./charmrun commSpeed +p4 100 100 $(TESTOPTS)
	./charmrun commSpeed +p8 100 100 $(TESTOPTS)
