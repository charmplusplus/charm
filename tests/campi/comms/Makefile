-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: comms

comms: comms.o
	$(CHARMC) -o comms comms.o

comms.o: comms.C
	$(CHARMC) -std=c++20 -c comms.C

clean:
	rm -f *.o comms charmrun conv-host ampirun

test: comms
	$(call run, ./comms +p1 +vp4)
	$(call run, ./comms +p2 +vp4)
	$(call run, ./comms +p2 +vp5)
	$(call run, ./comms +p2 +vp7)
