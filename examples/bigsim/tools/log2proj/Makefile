CHARMC=../../../../bin/charmc $(OPTS)

all: log2proj

log2proj: log2proj.o
	$(CHARMC) -o log2proj log2proj.o -lconv-bigsim-logs -ltrace-projections -language charm++  -lz

log2proj.o: log2proj.C  log2proj.decl.h
	$(CHARMC) -c log2proj.C

log2proj.decl.h: log2proj.ci
	$(CHARMC)  log2proj.ci

clean:
	rm -f *.o log2proj *~ moduleinit.C charmrun conv-host
	rm -rf *.decl.h *.def.h core *.log

test: log2proj
	./log2proj
