CHARMC=../../../bin/charmc $(OPTS)

OBJS=commbench.o overhead.o timer.o proc.o smputil.o pingpong.o ctxt.o

all: pgm

pgm: $(OBJS) 
	$(CHARMC) -o pgm $(OBJS) -language converse++

test: pgm
	./charmrun ./pgm +p4 $(TESTOPTS)

commbench.o: commbench.c
	$(CHARMC) commbench.c

overhead.o: overhead.c
	$(CHARMC) overhead.c

timer.o: timer.c
	$(CHARMC) timer.c

proc.o: proc.c
	$(CHARMC) proc.c

smputil.o: smputil.c
	$(CHARMC) smputil.c

pingpong.o: pingpong.c
	$(CHARMC) pingpong.c

ctxt.o: ctxt.c
	$(CHARMC) ctxt.c

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f pgm
	rm -f conv-host charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk

