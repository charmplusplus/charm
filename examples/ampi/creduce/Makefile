CHARMC=../../../bin/charmc $(OPTS)

ampi: test.C
	$(CHARMC) -c test.C
	$(CHARMC) -o pgm test.o -language ampi

mpi: test.C
	mpiCC -o pgm test.C $(OPTS)

clean:
	rm -f *.o *.mod pgm *~ conv-host charmrun

test: ampi
	./charmrun ./pgm +p1 +vp1
	./charmrun ./pgm +p1 +vp2
	./charmrun ./pgm +p1 +vp4
	./charmrun ./pgm +p2 +vp1
	./charmrun ./pgm +p2 +vp2
	./charmrun ./pgm +p2 +vp4

