-include ../common.mk
CHARMC=../../bin/charmc $(OPTS)

LINKLINE=$(CHARMC) -o check check.o -language converse++

all: check

check: check.o
	$(LINKLINE)

check.o: check.C
	$(CHARMC) -c check.C

headerpad: headerpad.o
	$(CHARMC) -language charm++ -o headerpad headerpad.o -tracemode recordreplay

headerpad.decl.h: headerpad.ci
	$(CHARMC)  headerpad.ci

headerpad.o: headerpad.C headerpad.decl.h headerpad.h
	$(CHARMC) -c headerpad.C


test: check
	$(call run, +p4 ./check 4)
	@echo tests if MACHINE_DEBUG is enabled in machine.h and fails if it is set
	@if test -f "debugLog.0"; then echo "MACHINE_DEBUG is enabled"; false; fi

clean:
	rm -f conv-host *.o *.bak *.log check.sts *~ charmrun charmrun.exe check.exe check.pdb check.ilk debugLog.*
