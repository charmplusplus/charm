OPTS=-g -Dcompile #-DCMK_OPTIMIZE=1 -DCMK_MEMCHECKS_OFF=1

CHARMBASE=../../../../
CHARMBIN=$(CHARMBASE)/bin
CHARMINC=$(CHARMBASE)/include

CHARMC=$(CHARMBIN)/charmc -I$(CHARMINC)/pose $(OPTS) 

LIBS=-Lstdc++ -lm

RedStorm: RoundRobin.o
	cp RoundRobin.o ../tmp

HypCubeArch: RoundRobin.o
	cp RoundRobin.o ../tmp

BlueGene: SLQ.o
	cp SLQ.o ../tmp

IB: SLQ_Switch.o
	cp SLQ_Switch.o ../tmp

OB: SLQ_Switch.o
	cp SLQ_Switch.o ../tmp

RoundRobin.o: RoundRobin.C RoundRobin.h
	$(CHARMC) -o RoundRobin.o -seq RoundRobin.C -I$(MACHINE)

SLQ.o: SLQ.C SLQ.h
	$(CHARMC) -o SLQ.o -seq SLQ.C -I$(MACHINE)

SLQ_Switch.o: SLQ_Switch.C SLQ_Switch.h
	$(CHARMC) -o SLQ_Switch.o -seq SLQ_Switch.C -I$(MACHINE)

clean:
	rm -f charmrun conv-host *.o
	rm -f *.def.h *.decl.h
	rm -f $(PGM) $(PGM).pure $(PGM).proj $(PGM).summary
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers
	rm -f *_sim.*
