CHARMC = ../../../../bin/charmc $(OPTS)
MODULES = -module commlib
#OPTS = -tracemode projections

OBJS = test.o benchmark.o

all: test benchmark

test: test.o
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
	rm -f *.decl.h *.def.h *.o test benchmark charmrun *~

new: clean all
