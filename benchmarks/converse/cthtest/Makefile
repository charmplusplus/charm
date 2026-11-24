-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

LINKLINE=$(CHARMC) -o cthtest pgm.o -language converse++

all: cthtest

cthtest: pgm.o
	$(LINKLINE)

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

test: cthtest
	$(call run, ./cthtest +p1 )


testp: cthtest
	$(call run, ./cthtest +p$(P))

clean:
	rm -f conv-host *.o cthtest *.bak cthtest.*.log cthtest.sts *~ charmrun charmrun.exe cthtest.exe cthtest.pdb cthtest.ilk
