OPTS=-g -Dcompile #-DCMK_OPTIMIZE=1 -DCMK_MEMCHECKS_OFF=1

include ../Makefile.common

LIBS=-Lstdc++ -lm

RedStorm: $(OBJDIR)/maxAvailBuffer.o

HypCubeArch: $(OBJDIR)/maxAvailBuffer.o  

BlueGene: $(OBJDIR)/maxAvailBufferBubbleVc.o

IB: $(OBJDIR)/maxAvailBufferSwitch.o

OB: $(OBJDIR)/outputBuffer.o

$(OBJDIR)/maxAvailBuffer.o: maxAvailBuffer.C maxAvailBuffer.h
	$(CHARMC) -o $@ -seq maxAvailBuffer.C -I$(MACHINE)

$(OBJDIR)/maxAvailBufferBubbleVc.o: maxAvailBufferBubbleVc.C maxAvailBufferBubbleVc.h
	$(CHARMC) -o $@ -seq maxAvailBufferBubbleVc.C -I$(MACHINE)

$(OBJDIR)/maxAvailBufferSwitch.o: maxAvailBufferSwitch.C maxAvailBufferSwitch.h
	$(CHARMC) -o $@ -seq maxAvailBufferSwitch.C -I$(MACHINE)
	
$(OBJDIR)/outputBuffer.o: outputBuffer.C outputBuffer.h
	$(CHARMC) -o $@ -seq outputBuffer.C -I$(MACHINE)

clean:
	rm -f charmrun conv-host *.o
	rm -f *.def.h *.decl.h
	rm -f $(PGM) $(PGM).pure $(PGM).proj $(PGM).summary
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers
	rm -f *_sim.*
