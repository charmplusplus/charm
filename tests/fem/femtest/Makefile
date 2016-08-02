-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS) 

all: pgm 

pgm: pgm.o
	$(CHARMC) -o pgm pgm.o -language fem

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

fpgm: fpgm.o
	$(CHARMC) -o fpgm fpgm.o -language femf

fpgm.o: fpgm.f90
	$(CHARMC) -c fpgm.f90

clean:
	rm -f pgm fpgm *.o conv-host charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk FEMVERSION

test: pgm
	$(call run, ./pgm +p1 +vp1 )
	$(call run, ./pgm +p1 +vp2 )
	$(call run, ./pgm +p2 +vp1 )
	$(call run, ./pgm +p2 +vp2 )
	$(call run, ./pgm +p2 +vp3 )
	$(call run, ./pgm +p2 +vp4 )
	-@rm -f fem_mesh_vp3_*
	$(call run, ./pgm +p1 +vp3 -write )
	$(call run, ./pgm +p1 +vp3 -read )
	rm -f fem_mesh_vp3_*
	-make testf

testf: fpgm
	$(call run, ./fpgm +p1 +vp1 )
	$(call run, ./fpgm +p1 +vp2 )
	$(call run, ./fpgm +p2 +vp1 )
	$(call run, ./fpgm +p2 +vp2 )
	$(call run, ./fpgm +p2 +vp3 )
	$(call run, ./fpgm +p2 +vp4 )
	-@rm -f fem_mesh_vp3_*
	$(call run, ./fpgm +p1 +vp3 -write )
	$(call run, ./fpgm +p1 +vp3 -read )
	rm -f fem_mesh_vp3_*

bgtest: pgm
	$(call run, ./pgm +p1 +vp1 +x1 +y1 +z1 )
	$(call run, ./pgm +p1 +vp2 +x1 +y1 +z1 )
	$(call run, ./pgm +p2 +vp1 +x2 +y1 +z1 )
	$(call run, ./pgm +p2 +vp2 +x2 +y1 +z1 )
	$(call run, ./pgm +p2 +vp3 +x2 +y1 +z1 )
	$(call run, ./pgm +p2 +vp4 +x2 +y1 +z1 )
	-@rm -f fem_mesh_vp3_*
	$(call run, ./pgm +p1 +vp3 -write +x3 +y1 +z1 )
	$(call run, ./pgm +p1 +vp3 -read  +x3 +y1 +z1 )
	rm -f fem_mesh_vp3_*

test-mig:
	rm -fr pgm pgm.o
	$(MAKE) OPTS="-DENABLE_MIG=1 -memory isomalloc -balancer RandCentLB" pgm
	$(call run, ./pgm +p2 )
