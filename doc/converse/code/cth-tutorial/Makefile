CHARMC?=../../../../bin/charmc $(OPTS)

LINKLINE=$(CHARMC) -o pgm pgm.o -language converse++

all: pgm

pgm: pgm.o
	$(LINKLINE)

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

test: pgm
	./charmrun ./pgm +p1 $(TESTOPTS)
#	-$(LINKLINE) -thread context && ./charmrun ./pgm +p1  $(TESTOPTS)&& ps -u `whoami`
#	-$(LINKLINE) -thread pthreads -lpthread && ./charmrun ./pgm +p1  $(TESTOPTS)&& ps -u `whoami`
#	-$(LINKLINE) -thread qt && ./charmrun ./pgm +p1  $(TESTOPTS)&& ps -u `whoami`
#	-$(LINKLINE) -thread uJcontext && ./charmrun ./pgm +p1  $(TESTOPTS)&& ps -u `whoami`

clean:
	rm -f conv-host *.o pgm *.bak pgm.*.log pgm.sts *~ charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk
