BIN=../../bin
CHARMC=$(BIN)/charmc $(OPTS)

all: $(BIN)/conv-host

$(BIN)/conv-host: conv-host.c 
	$(CHARMC) -c -seq -DSILENT conv-host.c
	$(CHARMC) -cp $(BIN) -seq -o conv-host-silent conv-host.o
	$(CHARMC) -c -seq -DNOTIFY conv-host.c
	$(CHARMC) -cp $(BIN) -seq -o conv-host-notify conv-host.o
	(if test "$$OSTYPE" = cygwin; then cp $(BIN)/conv-host-notify.exe $(BIN)/conv-host.exe; else cp $(BIN)/conv-host-notify $(BIN)/conv-host; fi)

clean:
	rm -f *.o conv-host conv-host-notify conv-host-silent
