OPTS=-g -Dcompile #-DCMK_OPTIMIZE=1 -DCMK_MEMCHECKS_OFF=1

include ../Makefile.common

LIBS=-Lstdc++ -lm

RedStorm: $(OBJDIR)/DirectionOrdered3D.o

HypCubeArch: $(OBJDIR)/HammingDistance.o

HypCubeArchOB: $(OBJDIR)/HammingDistance.o

BlueGene: $(OBJDIR)/TorusRouting.o

IB: $(OBJDIR)/UpDown.o

OB: $(OBJDIR)/UpDown.o

$(OBJDIR)/DirectionOrdered3D.o: DirectionOrdered3D.C DirectionOrdered3D.h
	$(CHARMC) -o $@ -seq DirectionOrdered3D.C -I$(MACHINE)

$(OBJDIR)/HammingDistance.o: HammingDistance.C HammingDistance.h
	$(CHARMC) -o $@ -seq HammingDistance.C -I$(MACHINE)

$(OBJDIR)/TorusRouting.o: TorusRouting.C TorusRouting.h
	$(CHARMC) -o $@ -seq TorusRouting.C -I$(MACHINE)

$(OBJDIR)/UpDown.o: UpDown.C UpDown.h
	$(CHARMC) -o $@ -seq UpDown.C -I$(MACHINE)

clean:
	rm -f charmrun conv-host *.o
	rm -f *.def.h *.decl.h
	rm -f $(PGM) $(PGM).pure $(PGM).proj $(PGM).summary
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers
	rm -f *_sim.*
