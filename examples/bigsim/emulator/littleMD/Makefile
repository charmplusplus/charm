-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

all: bgMD

bgMD: bgMD.o Handlers.o ../common.C
	$(CHARMC) -language bigsim -o bgMD $^

bgMD.o: bgMD.C bgMD.h Atom.h Cell.h SimParams.h
	$(CHARMC) -c bgMD.C

Handlers.o: bgMD.h Atom.h Cell.h SimParams.h Handlers.C
	$(CHARMC) -c Handlers.C

test: jacobi3D maxReduce
	$(call run, ./jacobi3D +p2  3 3 3 2 10)
	$(call run, ./maxReduce +p2 +cth3 +wth10)

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f bgMD
	rm -f conv-host charmrun
