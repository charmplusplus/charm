-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: self_send

self_send: self_send.o
	$(CHARMC) -language converse++ -o self_send self_send.o

self_send.o: self_send.cpp
	$(CHARMC) -language converse++ -c self_send.cpp

test: self_send
	$(call run, ./self_send +p1)
	$(call run, ./self_send +p1)

testp: self_send
	$(call run, ./self_send +p$(P))

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f self_send
	rm -f conv-host charmrun
