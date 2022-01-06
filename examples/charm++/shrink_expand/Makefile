-include ../../common.mk
CHARMC = ../../../bin/charmc

all: client

client : client.o
	$(CHARMC) -language c++ -seq -o client client.o -lccs-client

client.o : client.C
	$(CHARMC) -o client.o -c client.C

clean:
	rm -f *.o client *~
