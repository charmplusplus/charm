CHARMC=../../bin/charmc $(OPTS)

LINKLINE=$(CHARMC) -o check check.o -language converse++

all: check

check: check.o
	$(LINKLINE)

check.o: check.C
	$(CHARMC) -c check.C

test: check
	./charmrun +p1 check $(TESTOPTS)
	#The line below tests if MACHINE_DEBUG is enabled in machine.h and fails if it is set
	if test -f "debugLog.0"; then echo "MACHINE_DEBUG is enabled"; false; fi

clean:
	rm -f conv-host *.o *.bak *.log check.sts *~ charmrun
