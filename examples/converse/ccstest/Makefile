CHARMDIR=../../..
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)

all: server client

server: server.o
	$(CHARMC) -o $@ server.o -language converse++ 

client: client.o
	$(CHARMC) -o $@ client.o -seq -language c++ -lccs-client

server.o: server.c
	$(CHARMC) -c server.c -language converse++

client.o: client.C
	$(CHARMC) -c client.C -seq -language c++ 

clean:
	rm -f client server *.o conv-host charmrun
