BIN=../../bin
CHARMC=$(BIN)/charmc $(OPTS) -lm -I.. -L../libs

SHELL=/bin/sh

INCLUDED=../conv-mach.h ../conv-mach-opt.h \
	../ccs-server.C ../ccs-server.h ../ccs-auth.C ../ccs-auth.h \
	../sockRoutines.C ../sockRoutines.h

all: charmrun charmd charmd_faceless

charmrun: charmrun.C $(INCLUDED)
	$(CHARMC) -c -seq -DCMK_NOT_USE_CONVERSE=1 charmrun.C -o charmrun.o
	$(CHARMC) -cp $(BIN) -seq -language c++ -o charmrun charmrun.o

charmd: daemon.C daemon.h ../sockRoutines-seq.o
	$(CHARMC) -seq -c daemon.C -o daemon.o
	$(CHARMC) -cp $(BIN) -seq daemon.o ../sockRoutines-seq.o -o charmd

charmd_faceless: daemon.C daemon.h ../sockRoutines-seq.o
	$(CHARMC) -seq -DFACELESS -c daemon.C -o daemon-faceless.o
	$(CHARMC) -cp $(BIN) -seq -DFACELESS daemon-faceless.o ../sockRoutines-seq.o -o charmd_faceless;

clean:
	rm -f *.o charmrun
	rm -f charmd charmd_faceless
