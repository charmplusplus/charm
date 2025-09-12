-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: ping_ack

ping_ack: ping.o
	$(CHARMC) -language converse++ -o ping_ack ping.o

ping.o: ping.cpp
	$(CHARMC) -language converse++ -c ping.cpp

test: ping_ack
	$(call run, ./ping_ack +p2 )
 
testp: ping_ack
	$(call run, ./ping_ack +p$(P))

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f ping_ack
	rm -f conv-host charmrun
