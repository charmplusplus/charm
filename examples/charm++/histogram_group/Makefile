CHARMC	=../../../bin/charmc $(OPTS)
OPTS	= -O3 -g

all: histogram

histogram: pgm.o 
	$(CHARMC) -language charm++ -o histogram pgm.o

pgm.o: pgm.cc pgm.decl.h 
	$(CHARMC) -o pgm.o pgm.cc

pgm.decl.h: pgm.ci
	$(CHARMC) pgm.ci

clean:
	rm -f *.decl.h *.def.h *.o histogram charmrun
