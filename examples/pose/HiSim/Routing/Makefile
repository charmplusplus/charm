OPTS=-g -Dcompile #-DCMK_OPTIMIZE=1 -DCMK_MEMCHECKS_OFF=1

CHARMBASE=../../../../
CHARMBIN=$(CHARMBASE)/bin
CHARMINC=$(CHARMBASE)/include

CHARMC=$(CHARMBIN)/charmc -I$(CHARMINC)/pose $(OPTS) 

LIBS=-Lstdc++ -lm

RedStorm: DirectionOrdered3D.o
	cp DirectionOrdered3D.o ../tmp

HypCubeArch: HammingDistance.o
	cp HammingDistance.o ../tmp

BlueGene: TorusRouting.o
	cp TorusRouting.o ../tmp

IB: UpDown.o
	cp UpDown.o ../tmp

OB: UpDown.o
	cp UpDown.o ../tmp

DirectionOrdered3D.o: DirectionOrdered3D.C DirectionOrdered3D.h
	$(CHARMC) -o DirectionOrdered3D.o -seq DirectionOrdered3D.C -I$(MACHINE)

HammingDistance.o: HammingDistance.C HammingDistance.h
	$(CHARMC) -o HammingDistance.o -seq HammingDistance.C -I$(MACHINE)

TorusRouting.o: TorusRouting.C TorusRouting.h
	$(CHARMC) -o TorusRouting.o -seq TorusRouting.C -I$(MACHINE)

UpDown.o: UpDown.C UpDown.h
	$(CHARMC) -o UpDown.o -seq UpDown.C -I$(MACHINE)

clean:
	rm -f charmrun conv-host *.o
	rm -f *.def.h *.decl.h
	rm -f $(PGM) $(PGM).pure $(PGM).proj $(PGM).summary
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers
	rm -f *_sim.*
