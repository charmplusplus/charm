-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: integrate

integrate: pgm.o
	$(CHARMC) pgm.o  -o integrate -language charm++

pgm.o : pgm.C pgm.h pgm.ci
	$(CHARMC) -c pgm.ci pgm.C

clean:
	rm -f integrate *.o conv-host *.decl.h *.def.h *~ charmrun charmrun.exe integrate.exe integrate.pdb integrate.ilk

test: integrate
	$(call run, ./integrate +p4 1000000 100)

testp: integrate
	$(call run, ./integrate +p$(P) 1000000 $$(( $(P) * 10 )) )
