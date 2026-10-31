CHARMDIR=../../..
CHARMC=$(CHARMDIR)/bin/charmc $(OPTS)

all: server client

server: server.o
	$(CHARMC) -o $@ server.o -language converse++

client: client.o
	g++ -o $@ client.o -I$(CHARMDIR)/include -L$(CHARMDIR)/lib -lccs-client

server.o: server.C
	$(CHARMC) -c server.C -language converse++

# The headers need C++11 or later (reconverse's converse.h needs C++17);
# Apple clang defaults to C++98.
client.o: client.C
	g++ -std=gnu++17 -c client.C -I$(CHARMDIR)/include

clean:
	rm -f client server *.o conv-host charmrun
