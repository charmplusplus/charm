-include ../../common.mk
CHARMC := ../../../bin/charmc

all: qdbench

qdbench: qdbench.o
	$(CHARMC) $(OPTS) -o $@ qdbench.o

qdbench.o: qdbench.C qdbench.decl.h
	$(CHARMC) $(OPTS) -c qdbench.C

qdbench.decl.h: qdbench.ci
	$(CHARMC) qdbench.ci

test: all
	$(call run, ./qdbench +p2)
	$(call run, ./qdbench +p4)

testp: all
	$(call run, ./qdbench +p$(P))

smptest: all
	$(call run, ./qdbench +p4 ++ppn 2)

clean:
	rm -f *.decl.h *.def.h *.o qdbench charmrun
