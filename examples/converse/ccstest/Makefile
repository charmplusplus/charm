CHARMDIR=../../..
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)

all: server client

server: server.o
	$(CHARMC) -o $@ server.o -language converse++ 

client: client.o
	g++ -o $@ client.o -I$(CHARMDIR)/include -L$(CHARMDIR)/lib -lccs-client

server.o: server.C
	$(CHARMC) -c server.C -language converse++

client.o: client.C
	g++ -c client.C -I$(CHARMDIR)/include

clean:
	rm -f client server *.o conv-host charmrun
