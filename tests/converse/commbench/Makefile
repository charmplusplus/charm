-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

OBJS=memoryAccess.o commbench.o overhead.o timer.o proc.o smputil.o pingpong.o \
    flood.o broadcast.o reduction.o ctxt.o

all: pgm

pgm: $(OBJS) 
	$(CHARMC) -o pgm $(OBJS) -language converse++

test: pgm
	$(call run, ./pgm +p4 )

memoryAccess.o: memoryAccess.c
	$(CHARMC) memoryAccess.c

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

flood.o: flood.c
	$(CHARMC) flood.c

broadcast.o: broadcast.c
	$(CHARMC) broadcast.c

reduction.o: reduction.c
	$(CHARMC) reduction.c

ctxt.o: ctxt.c
	$(CHARMC) ctxt.c

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f pgm
	rm -f conv-host charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk

