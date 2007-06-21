BIN=../../bin
CHARMC=$(BIN)/charmc $(OPTS)

SHELL=/bin/sh

INCLUDED=../conv-mach.h ../conv-mach-opt.h \
	../ccs-server.c ../ccs-server.h ../ccs-auth.c ../ccs-auth.h \
	../sockRoutines.c ../sockRoutines.h

all: charmrun-silent charmrun-notify charmd charmd-faceless post-config

charmrun-silent.o: charmrun.c $(INCLUDED)
	$(CHARMC) -seq -DSILENT -c charmrun.c -o charmrun-silent.o
charmrun-silent: charmrun-silent.o
	$(CHARMC) -cp $(BIN) charmrun-silent.o -o charmrun-silent
charmrun-notify.o: charmrun.c $(INCLUDED)
	$(CHARMC) -seq -DNOTIFY -c charmrun.c -o charmrun-notify.o
charmrun-notify: charmrun-notify.o
	$(CHARMC) -cp $(BIN) charmrun-notify.o -o charmrun-notify
post-config: charmrun-notify charmrun-silent
	(cd ..; CHARMINC=.; . ./conv-config.sh; cp ../bin/charmrun-notify$$CMK_POST_EXE ../bin/charmrun$$CMK_POST_EXE)

daemon.o: daemon.c daemon.h
	$(CHARMC) -seq -c daemon.c -o daemon.o
daemon-faceless.o: daemon.c daemon.h
	$(CHARMC) -DFACELESS -c daemon.c -o daemon-faceless.o

charmd: daemon.o ../sockRoutines-seq.o
	$(CHARMC) -cp $(BIN) -seq daemon.o ../sockRoutines-seq.o -o charmd
charmd-faceless: daemon-faceless.o ../sockRoutines-seq.o
	$(CHARMC) -cp $(BIN) daemon-faceless.o ../sockRoutines-seq.o -o charmd_faceless;

clean:
	rm -f *.o charmrun charmrun-notify charmrun-silent
	rm -f charmd charmd_faceless
