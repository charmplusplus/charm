CHARMC=../../../bin/charmc $(OPTS)

all: pgm

ampi: clean pgm

pgm: test.C
	$(CHARMC) -c test.C
	$(CHARMC) -o pgm test.o -language ampi

mpi: test.C
	mpiCC -o pgm test.C $(OPTS)

clean:
	rm -f *.o *.mod pgm *~ conv-host charmrun

test: pgm
	./charmrun ./pgm +p1 +vp1
	./charmrun ./pgm +p1 +vp2
	./charmrun ./pgm +p1 +vp4
	./charmrun ./pgm +p2 +vp1
	./charmrun ./pgm +p2 +vp2
	./charmrun ./pgm +p2 +vp4

bgtest: ampi
	./charmrun ./pgm +p1 +vp1 +x1 +y1 +z1
	./charmrun ./pgm +p1 +vp2 +x1 +y1 +z1
	./charmrun ./pgm +p1 +vp4 +x1 +y1 +z1
	./charmrun ./pgm +p2 +vp1 +x2 +y1 +z1
	./charmrun ./pgm +p2 +vp2 +x2 +y1 +z1
	./charmrun ./pgm +p2 +vp4 +x2 +y1 +z1

