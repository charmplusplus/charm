-include ../../../common.mk

CHARMC = ../../../../bin/charmc $(OPTS)

all: memcontract

memcontract: memcontract.o
	$(CHARMC) -language charm++ -o $@ $<

memcontract.decl.h memcontract.def.h: memcontract.ci
	$(CHARMC) memcontract.ci

memcontract.o: memcontract.C memcontract.decl.h ../../../../src/ck-ldb/LBMemoryContract.h
	$(CHARMC) -c memcontract.C

test: memcontract
	$(call run, +p1 ./memcontract)

testp: test

smptest: test

clean:
	rm -f *.decl.h *.def.h *.o charmrun charmrun.exe memcontract \
	  memcontract.exe memcontract.pdb memcontract.ilk
