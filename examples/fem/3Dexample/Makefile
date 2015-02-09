-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS) 

all: pgm

pgm: pgm.o
	$(CHARMC) -o pgm pgm.o -language fem -module netfem #-memory paranoid 

pgm.o: pgm.C
	$(CHARMC) -g -I../../../src/libs/ck-libs/fem -c pgm.C

test: pgm
	$(call run, ./pgm +vp4 +p2)

bgtest: pgm
	$(call run, ./pgm +vp4 +p2 +x2 +y2 +z1)

clean:
	rm -rf pgm fpgm *.o conv-host charmrun NetFEM Paraview* ParaView* *~ pvClient.out README
