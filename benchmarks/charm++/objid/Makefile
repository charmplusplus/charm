-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: objid_bench

objid_bench: objid_bench.o
	$(CHARMC) -language charm++ -o objid_bench objid_bench.o

objid_bench.decl.h: objid_bench.ci
	$(CHARMC) objid_bench.ci

objid_bench.o: objid_bench.C objid_bench.decl.h
	$(CHARMC) -c objid_bench.C

test: all
	$(call run, +p4 ./objid_bench -k bounded -n 4000 -r 2)
	$(call run, +p4 ./objid_bench -k unbounded -n 4000 -r 2)

clean:
	rm -f *.decl.h *.def.h *.o objid_bench charmrun
