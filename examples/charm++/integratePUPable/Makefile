-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: integratePUPable

integratePUPable: pgm.o
	$(CHARMC) pgm.o  -o integratePUPable -language charm++

pgm.o : pgm.C pgm.h pgm.decl.h
	$(CHARMC) -c pgm.C

pgm.decl.h: pgm.ci
	$(CHARMC) -c pgm.ci

clean:
	rm -f integratePUPable *.o conv-host *.decl.h *.def.h *~ charmrun charmrun.exe integratePUPable.exe integratePUPable.pdb integratePUPable.ilk

test: integratePUPable
	$(call run, ./integratePUPable +p2 1000000 100 )

testp: integratePUPable
	$(call run, ./integratePUPable +p$(P) 1000000 $$(( $(P) * 10 )) )
