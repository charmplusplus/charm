CHARMC=../../../bin/charmc $(OPTS) 

all: pgm 

pgm: pgm.o
	$(CHARMC) -o pgm pgm.o -language fem -module netfem -lmoduleifem

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

clean:
	rm -f pgm fpgm *.o conv-host charmrun

test: pgm
	./charmrun ./pgm +p1 +vp1
	./charmrun ./pgm +p1 +vp2
	./charmrun ./pgm +p2 +vp1
	./charmrun ./pgm +p2 +vp2
	./charmrun ./pgm +p2 +vp3
	./charmrun ./pgm +p2 +vp4

bgtest: pgm
	./charmrun ./pgm +p1 +vp1 +x1 +y1 +z1
	./charmrun ./pgm +p1 +vp2 +x1 +y1 +z1
	./charmrun ./pgm +p2 +vp1 +x2 +y1 +z1
	./charmrun ./pgm +p2 +vp2 +x2 +y1 +z1
	./charmrun ./pgm +p2 +vp3 +x2 +y1 +z1
	./charmrun ./pgm +p2 +vp4 +x2 +y1 +z1
