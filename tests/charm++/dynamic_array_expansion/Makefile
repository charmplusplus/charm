-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: dynamical_insertion

dynamical_insertion: dynamical_insertion.decl.h dynamical_insertion.C
	$(CHARMC) dynamical_insertion.C -o dynamical_insertion 

dynamical_insertion.decl.h: dynamical_insertion.ci
	$(CHARMC) dynamical_insertion.ci

test: dynamical_insertion
	$(call run, +p1 ./dynamical_insertion)
	$(call run, +p2 ./dynamical_insertion)
	$(call run, +p4 ./dynamical_insertion)

testp: dynamical_insertion
	$(call run, +p$(P) ./dynamical_insertion)

smptest: dynamical_insertion
	$(call run, +p2 ./dynamical_insertion)
	$(call run, +p4 ./dynamical_insertion)

clean:
	rm -rf *.decl.h *.def.h dynamical_insertion charmrun
