CHARMC = ../../../../bin/charmc $(OPTS)
MODULES = -module comlib
#OPTS = -tracemode projections

OBJS = test.o benchmark.o

all: pgm benchmark

test: pgm
	./charmrun ./pgm +p4 10

bgtest: pgm
	./charmrun ./pgm +p2 10 +x2 +y2 +z1

pgm: test.o
	$(CHARMC) -language charm++ -o $@ test.o $(MODULES) 

test.decl.h: test.ci
	$(CHARMC) test.ci

test.o: test.C test.decl.h test.h
	$(CHARMC) -c test.C

benchmark: benchmark.o
	$(CHARMC) -language charm++ -o $@ benchmark.o $(MODULES)

benchmark.decl.h: benchmark.ci
	$(CHARMC) benchmark.ci

benchmark.o: benchmark.C benchmark.decl.h benchmark.h
	$(CHARMC) -c benchmark.C

clean:
	rm -f *.decl.h *.def.h *.o pgm benchmark charmrun *~

new: clean all
