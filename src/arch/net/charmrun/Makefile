BIN=../../bin
CHARMC=$(BIN)/charmc $(OPTS)

SHELL=/bin/sh

$(BIN)/charmrun: charmrun.c 
	$(CHARMC) -c -seq -DSILENT charmrun.c
	$(CHARMC) -cp $(BIN) -seq -o charmrun-silent charmrun.o
	$(CHARMC) -c -seq -DNOTIFY charmrun.c
	$(CHARMC) -cp $(BIN) -seq -o charmrun-notify charmrun.o
	if test "$$OSTYPE" = cygwin; then cp $(BIN)/charmrun-notify.exe $(BIN)/charmrun.exe; else cp $(BIN)/charmrun-notify $(BIN)/charmrun; fi

clean:
	rm -f *.o charmrun charmrun-notify charmrun-silent
