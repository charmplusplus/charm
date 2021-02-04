-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: rings

rings:  rings.o
	$(CHARMC) rings.o  -o rings -language charm++

rings.o : rings.C rings.h rings.ci
	$(CHARMC) -c rings.ci rings.C

test: rings
	$(call run, +p4 ./rings )

testp: rings
	$(call run, +p$(P) ./rings )

clean:
	rm -f conv-host rings *.o *.decl.h *.def.h *~ charmrun charmrun.exe rings.exe rings.pdb rings.ilk
