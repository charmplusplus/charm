BIN=../../bin
CHARMC=$(BIN)/charmc $(OPTS) -lm -I..

SHELL=/bin/sh

INCLUDED=../conv-mach.h ../conv-mach-opt.h \
	../ccs-server.c ../ccs-server.h ../ccs-auth.c ../ccs-auth.h \
	../sockRoutines.c ../sockRoutines.h

all: charmrun-notify charmrun-silent charmd charmd_faceless

charmrun-silent: charmrun.C $(INCLUDED)
	$(CHARMC) -c -seq -DCMK_NOT_USE_CONVERSE=1 -DSILENT charmrun.C -o charmrun-silent.o
	$(CHARMC) -cp $(BIN) -seq -language c++ -o charmrun-silent charmrun-silent.o

charmrun-notify: charmrun.C $(INCLUDED)
	$(CHARMC) -c -seq -DCMK_NOT_USE_CONVERSE=1 -DNOTIFY charmrun.C
	$(CHARMC) -cp $(BIN) -seq -language c++ -o charmrun-notify charmrun.o
	(cd ..; CHARMINC=.; . ./conv-config.sh; cp ../bin/charmrun-notify$$CMK_POST_EXE ../bin/charmrun$$CMK_POST_EXE)

charmd: daemon.c daemon.h ../sockRoutines-seq.o
	$(CHARMC) -seq -c daemon.c -o daemon.o
	$(CHARMC) -cp $(BIN) -seq daemon.o ../sockRoutines-seq.o -o charmd

charmd_faceless: daemon.c daemon.h ../sockRoutines-seq.o
	$(CHARMC) -seq -DFACELESS -c daemon.c -o daemon-faceless.o
	$(CHARMC) -cp $(BIN) -seq -DFACELESS daemon-faceless.o ../sockRoutines-seq.o -o charmd_faceless;

clean:
	rm -f *.o charmrun charmrun-notify charmrun-silent
	rm -f charmd charmd_faceless
