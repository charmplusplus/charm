OPTS=-g -Dcompile #-DCMK_OPTIMIZE=1 -DCMK_MEMCHECKS_OFF=1

CHARMBASE=../../../../
CHARMBIN=$(CHARMBASE)/bin
CHARMINC=$(CHARMBASE)/include

CHARMC=$(CHARMBIN)/charmc -I$(CHARMINC)/pose $(OPTS) 

LIBS=-Lstdc++ -lm


RedStorm: Mesh3D.o
	cp Mesh3D.o ../tmp

HypCubeArch: HyperCube.o
	cp HyperCube.o ../tmp

HypCubeArchOB: HyperCube.o
	cp HyperCube.o ../tmp

BlueGene: Mesh3D.o
	cp Mesh3D.o ../tmp

IB: FatTree.o
	cp FatTree.o ../tmp

OB: FatTree.o
	cp FatTree.o ../tmp

HyperCube.o: HyperCube.C HyperCube.h
	$(CHARMC) -o HyperCube.o -seq HyperCube.C -I$(MACHINE)

Mesh3D.o: Mesh3D.C Mesh3D.h
	$(CHARMC) -o Mesh3D.o -seq Mesh3D.C -I$(MACHINE)

FatTree.o: FatTree.C FatTree.h
	$(CHARMC) -o FatTree.o -seq FatTree.C -I$(MACHINE)

clean:
	rm -f charmrun conv-host *.o
	rm -f *.def.h *.decl.h
	rm -f $(PGM) $(PGM).pure $(PGM).proj $(PGM).summary
	rm -f gmon.out #*#
	rm -f core *~
	rm -f TAGS *.headers
	rm -f *_sim.*
