CHARMC=../../bin/charmc $(OPTS)

LINKLINE=$(CHARMC) -o check check.o -language converse++

check: check.o
	$(LINKLINE)

check.o: check.C
	$(CHARMC) -c check.C

test: check
	./charmrun +p1 check

clean:
	rm -f conv-host *.o check.* *.bak *.log check.sts *~ charmrun
