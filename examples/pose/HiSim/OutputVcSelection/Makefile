OPTS=-g -Dcompile #-DCMK_OPTIMIZE=1 -DCMK_MEMCHECKS_OFF=1

CHARMBASE=../../../../
CHARMBIN=$(CHARMBASE)/bin
CHARMINC=$(CHARMBASE)/include

CHARMC=$(CHARMBIN)/charmc -I$(CHARMINC)/pose $(OPTS) 

LIBS=-Lstdc++ -lm

RedStorm: maxAvailBuffer.o
	cp maxAvailBuffer.o ../tmp

HypCubeArch: maxAvailBuffer.o
	cp maxAvailBuffer.o ../tmp

BlueGene: maxAvailBufferBubbleVc.o
	cp maxAvailBufferBubbleVc.o ../tmp

IB: maxAvailBufferSwitch.o
	cp maxAvailBufferSwitch.o ../tmp

OB: outputBuffer.o
	cp outputBuffer.o ../tmp

maxAvailBuffer.o: maxAvailBuffer.C maxAvailBuffer.h
	$(CHARMC) -o maxAvailBuffer.o -seq maxAvailBuffer.C -I$(MACHINE)

maxAvailBufferBubbleVc.o: maxAvailBufferBubbleVc.C maxAvailBufferBubbleVc.h
	$(CHARMC) -o maxAvailBufferBubbleVc.o -seq maxAvailBufferBubbleVc.C -I$(MACHINE)

maxAvailBufferSwitch.o: maxAvailBufferSwitch.C maxAvailBufferSwitch.h
	$(CHARMC) -o maxAvailBufferSwitch.o -seq maxAvailBufferSwitch.C -I$(MACHINE)
	
outputBuffer.o: outputBuffer.C outputBuffer.h
	$(CHARMC) -o outputBuffer.o -seq outputBuffer.C -I$(MACHINE)

clean:
	rm -f charmrun conv-host *.o
	rm -f *.def.h *.decl.h
	rm -f $(PGM) $(PGM).pure $(PGM).proj $(PGM).summary
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers
	rm -f *_sim.*
