BIN=../../bin
CHARMC=$(BIN)/charmc $(OPTS)

SHELL=/bin/sh

INCLUDED=../conv-mach.h ../conv-mach-opt.h \
	../ccs-server.c ../ccs-server.h ../ccs-auth.c ../ccs-auth.h \
	../sockRoutines.c ../sockRoutines.h

$(BIN)/charmrun: charmrun.c $(INCLUDED)
	$(CHARMC) -c -seq -DSILENT charmrun.c
	$(CHARMC) -cp $(BIN) -seq -o charmrun-silent charmrun.o
	$(CHARMC) -c -seq -DNOTIFY charmrun.c
	$(CHARMC) -cp $(BIN) -seq -o charmrun-notify charmrun.o
	if test "$$OSTYPE" = cygwin; then cp $(BIN)/charmrun-notify.exe $(BIN)/charmrun.exe; else cp $(BIN)/charmrun-notify $(BIN)/charmrun; fi

clean:
	rm -f *.o charmrun charmrun-notify charmrun-silent
