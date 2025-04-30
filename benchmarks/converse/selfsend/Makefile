-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: self_send

self_send: self_send.o
	$(CHARMC) -language converse++ -o self_send self_send.o

self_send.o: self_send.cpp
	$(CHARMC) -language converse++ -c self_send.cpp

test: self_send
	$(call run, ./self_send +p4)
	$(call run, ./self_send +p6)

testp: self_send
	$(call run, ./self_send +p$(P))
	# $(call run, ./self_send +p$(P) 100 1024) # Example with args if needed

clean:
	rm -f core *.cpm.h
	rm -f TAGS *.o
	rm -f self_send
	rm -f conv-host charmrun
