OPTS=-g #-DCMK_OPTIMIZE=1 -DCMK_MEMCHECKS_OFF=1

include ../Makefile.common

CHARMC=$(CHARMBIN)/charmc -I$(CHARMINC)/pose $(OPTS) -Dcompile

LIBS=-Lstdc++ -lm

RedStorm: $(OBJDIR)/RoundRobin.o

HypCubeArch: $(OBJDIR)/RoundRobin.o

HypCubeArchOB: $(OBJDIR)/outputBufferIn.o

BlueGene: $(OBJDIR)/SLQ.o

IB: $(OBJDIR)/SLQ_Switch.o

#OB: $(OBJDIR)/SLQ_Switch.o
OB: $(OBJDIR)/outputBufferIn.o

$(OBJDIR)/RoundRobin.o: RoundRobin.C RoundRobin.h
	$(CHARMC) -o $@ -seq RoundRobin.C -I$(MACHINE)

$(OBJDIR)/SLQ.o: SLQ.C SLQ.h
	$(CHARMC) -o $@ -seq SLQ.C -I$(MACHINE)

$(OBJDIR)/SLQ_Switch.o: SLQ_Switch.C SLQ_Switch.h
	$(CHARMC) -o $@ -seq SLQ_Switch.C -I$(MACHINE)

$(OBJDIR)/outputBufferIn.o: outputBufferIn.C outputBufferIn.h
	$(CHARMC) -o $@ -seq outputBufferIn.C -I$(MACHINE)

clean:
	rm -f charmrun conv-host *.o
	rm -f *.def.h *.decl.h
	rm -f $(PGM) $(PGM).pure $(PGM).proj $(PGM).summary
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers
	rm -f *_sim.*
