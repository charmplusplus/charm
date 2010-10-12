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
	rm -f pgm fpgm *.o conv-host charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk

test: pgm
	./charmrun ./pgm +p1 +vp1 $(TESTOPTS)
	./charmrun ./pgm +p1 +vp2 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp1 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp2 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp3 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp4 $(TESTOPTS)
	-@rm -f fem_mesh_vp3_*
	./charmrun ./pgm +vp3 -write $(TESTOPTS)
	./charmrun ./pgm +vp3 -read $(TESTOPTS)
	rm -f fem_mesh_vp3_*
	-make testf

testf: fpgm
	./charmrun ./fpgm +p1 +vp1 $(TESTOPTS)
	./charmrun ./fpgm +p1 +vp2 $(TESTOPTS)
	./charmrun ./fpgm +p2 +vp1 $(TESTOPTS)
	./charmrun ./fpgm +p2 +vp2 $(TESTOPTS)
	./charmrun ./fpgm +p2 +vp3 $(TESTOPTS)
	./charmrun ./fpgm +p2 +vp4 $(TESTOPTS)
	-@rm -f fem_mesh_vp3_*
	./charmrun ./fpgm +vp3 -write $(TESTOPTS)
	./charmrun ./fpgm +vp3 -read $(TESTOPTS)
	rm -f fem_mesh_vp3_*

bgtest: pgm
	./charmrun ./pgm +p1 +vp1 +x1 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +p1 +vp2 +x1 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp1 +x2 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp2 +x2 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp3 +x2 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp4 +x2 +y1 +z1 $(TESTOPTS)
	-@rm -f fem_mesh_vp3_*
	./charmrun ./pgm +vp3 -write +x3 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +vp3 -read  +x3 +y1 +z1 $(TESTOPTS)
	rm -f fem_mesh_vp3_*

test-mig:
	rm -fr pgm pgm.o
	$(MAKE) OPTS="-DENABLE_MIG=1 -memory isomalloc -balancer RandCentLB" pgm
	./charmrun ./pgm +p2 $(TESTOPTS)
