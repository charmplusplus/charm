CHARMC=../../../../bin/charmc $(OPTS) 

all: pgm fpgm

pgm: pgm.o
	$(CHARMC) -o pgm pgm.o -language fem

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

fpgm: fpgm.o
	$(CHARMC) -o fpgm fpgm.o -language femf

fpgm.o: fpgm.f90
	$(CHARMC) -c fpgm.f90

clean:
	rm -f pgm fpgm *.o conv-host charmrun

test: pgm
	./charmrun ./pgm +p1 +vp1
	./charmrun ./pgm +p1 +vp2
	./charmrun ./pgm +p2 +vp1
	./charmrun ./pgm +p2 +vp2
	./charmrun ./pgm +p2 +vp3
	./charmrun ./pgm +p2 +vp4
	-@rm -f fem_mesh_vp3_*
	./charmrun ./pgm +vp3 -write
	./charmrun ./pgm +vp3 -read
	rm -f fem_mesh_vp3_*

test-mig:
	rm -fr pgm pgm.o
	make OPTS="-DENABLE_MIG=1 -memory isomalloc -balancer RandCentLB" pgm
	./charmrun ./pgm +p2
