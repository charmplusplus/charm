CHARMC=../../../bin/charmc $(OPTS)

LINKLINE=$(CHARMC) -o pgm pgm.o -language converse++

pgm: pgm.o
	$(LINKLINE)

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

test: pgm
	./charmrun pgm +p1
	-$(LINKLINE) -thread context && ./charmrun pgm +p1
	-$(LINKLINE) -thread pthreads -lpthread && ./charmrun pgm +p1
	-$(LINKLINE) -thread qt && ./charmrun pgm +p1
	-$(LINKLINE) -thread uJcontext && ./charmrun pgm +p1

clean:
	rm -f conv-host *.o pgm *.bak pgm.*.log pgm.sts *~ charmrun
