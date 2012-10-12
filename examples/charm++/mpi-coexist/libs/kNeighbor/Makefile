include ../../Makefile.in
CHARMC:=../../${CHARMC}

all: libmodulekNeighbor.a

libmodulekNeighbor.a: kNeighbor.o
	$(CHARMC) -o libmodulekNeighbor.a kNeighbor.o

kNeighbor.decl.h: kNeighbor.ci
	$(CHARMC) kNeighbor.ci

kNeighbor.o: kNeighbor.C kNeighbor.decl.h
	$(CHARMC) -c kNeighbor.C

clean:
	rm -f *.decl.h *.def.h conv-host *.a *.o charmrun *~ 
