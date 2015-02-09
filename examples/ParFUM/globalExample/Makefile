-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS) 

all: pgm

pgm: pgm.o	
	$(CHARMC) -o glbEx pgm.o -language ParFUM -module netfem -module CommonLBs #-memory paranoid

pgm.o: pgm.C
	$(CHARMC) -I../../../src/libs/ck-libs/ParFUM -c pgm.C #-DLOAD_BALANCE

test: pgm
	$(call run, ./glbEx +vp4 +p2)

bgtest: pgm
	$(call run, ./glbEx +vp4 +p2 +x2 +y2 +z1)

clean:
	rm -rf pgm glbEx fpgm *.o *sts *log *projrc *sum charmrun NetFEM Paraview* ParaView* *~
