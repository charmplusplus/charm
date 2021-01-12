-include ../../common.mk
# modify the path of CHARMC
CHARMC=../../../bin/charmc $(OPTS)

all: integrateArray

integrateArray: pgm.o
	$(CHARMC) pgm.o  -o integrateArray -language charm++

test: integrateArray
	$(call run, +p4 ./integrateArray 1000000 100 )

testp: integrateArray
	$(call run, ./integrateArray +p$(P) 1000000 $$(( $(P) * 10 )) )

# compile program with trace projections - performance trace library
pgm_prof: pgm.o
	$(CHARMC) pgm.o  -o pgm_prof -language charm++ -tracemode projections

pgm.o : pgm.C pgm.h pgm.def.h
	$(CHARMC) -c pgm.C

pgm.decl.h pgm.def.h: pgm.ci
	$(CHARMC)  pgm.ci

clean:
	rm -f integrateArray pgm_prof *.o conv-host *.decl.h *.def.h *~ charmrun *.log *.sts core charmrun.exe integrateArray.exe integrateArray.pdb integrateArray.ilk
