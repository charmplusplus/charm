BIN=../../bin
CHARMC=$(BIN)/charmc $(OPTS)

all: conv-host

conv-host: conv-host.c
	$(CHARMC) -c -seq -DSILENT conv-host.c
	$(CHARMC) -cp $(BIN) -seq -o conv-host-silent conv-host.o
	$(CHARMC) -c -seq -DNOTIFY conv-host.c
	$(CHARMC) -cp $(BIN) -seq -o conv-host-notify conv-host.o
	cp conv-host-notify conv-host
	cp conv-host $(BIN)/conv-host

clean:
	rm -f *.o conv-host conv-host-notify conv-host-silent
