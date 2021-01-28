-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: piArray

piArray: pgm.o
	$(CHARMC) pgm.o  -o piArray -language charm++

test: piArray
	$(call run, +p4 ./piArray 100000 100 )

testp: piArray
	$(call run, +p$(P) ./piArray $$(( $(P) * 25000 )) $$(( $(P) * 25 )) )

pgm.o : pgm.C pgm.h pgm.decl.h
	$(CHARMC) -c pgm.C

pgm.decl.h: pgm.ci
	$(CHARMC) pgm.ci

clean:
	rm -f piArray *.o conv-host *.decl.h *.def.h *~ charmrun charmrun.exe piArray.exe piArray.pdb piArray.ilk
