BIN=../../bin
CHARMC=$(BIN)/charmc $(OPTS) -lm -I..

SHELL=/bin/sh

INCLUDED=../conv-mach.h ../conv-mach-opt.h \
	../ccs-server.c ../ccs-server.h ../ccs-auth.c ../ccs-auth.h \
	../sockRoutines.c ../sockRoutines.h

all: charmrun charmd charmd_faceless

charmrun: charmrun.C $(INCLUDED)
	$(CHARMC) -c -seq -DCMK_NOT_USE_CONVERSE=1 charmrun.C -o charmrun.o
	$(CHARMC) -cp $(BIN) -seq -language c++ -o charmrun charmrun.o

charmd: daemon.c daemon.h ../sockRoutines-seq.o
	$(CHARMC) -seq -c daemon.c -o daemon.o
	$(CHARMC) -cp $(BIN) -seq daemon.o ../sockRoutines-seq.o -o charmd

charmd_faceless: daemon.c daemon.h ../sockRoutines-seq.o
	$(CHARMC) -seq -DFACELESS -c daemon.c -o daemon-faceless.o
	$(CHARMC) -cp $(BIN) -seq -DFACELESS daemon-faceless.o ../sockRoutines-seq.o -o charmd_faceless;

clean:
	rm -f *.o charmrun
	rm -f charmd charmd_faceless
