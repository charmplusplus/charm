BIN=../../bin
CHARMC=$(BIN)/charmc $(OPTS)

SHELL=/bin/sh

INCLUDED=../conv-mach.h ../conv-mach-opt.h \
	../ccs-server.c ../ccs-server.h ../ccs-auth.c ../ccs-auth.h \
	../sockRoutines.c ../sockRoutines.h

all: charmrun-silent charmd

charmrun-silent: charmrun-notify

charmrun-notify: charmrun.c $(INCLUDED)
	$(CHARMC) -c -seq -DSILENT charmrun.c
	$(CHARMC) -cp $(BIN) -seq -o charmrun-silent charmrun.o
	$(CHARMC) -c -seq -DNOTIFY charmrun.c
	$(CHARMC) -cp $(BIN) -seq -o charmrun-notify charmrun.o
	(cd ..; CHARMINC=.; . ./conv-config.sh; cp ../bin/charmrun-notify$$CMK_POST_EXE ../bin/charmrun$$CMK_POST_EXE)

charmd: charmd_faceless

charmd_faceless: daemon.c daemon.h ../sockRoutines-seq.o
	$(CHARMC) -cp $(BIN) -seq daemon.c ../sockRoutines-seq.o -o charmd
	$(CHARMC) -cp $(BIN) -DFACELESS -seq daemon.c ../sockRoutines-seq.o -o charmd_faceless;

clean:
	rm -f *.o charmrun charmrun-notify charmrun-silent
	rm -f charmd charmd_faceless
