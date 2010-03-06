CHARMC=../../../bin/ampicxx $(OPTS)

all: pgm

pgm: test.o
	$(CHARMC) -o pgm test.o 

test.o: test.C
	$(CHARMC) -c test.C

#
#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod pgm *~ conv-host charmrun test.o charmrun.exe pgm.exe pgm.pdb pgm.ilk

test: pgm
	./charmrun ./pgm +p1 +vp1 $(TESTOPTS)
	./charmrun ./pgm +p1 +vp4 $(TESTOPTS)
	./charmrun ./pgm +p1 +vp20 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp1 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp4 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp20 $(TESTOPTS)

bgtest: pgm
	./charmrun ./pgm +p1 +vp1 +x1 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +p1 +vp2 +x1 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +p1 +vp4 +x1 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp2 +x2 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp4 +x2 +y1 +z1 $(TESTOPTS)
	./charmrun ./pgm +p2 +vp1 +x2 +y1 +z1 $(TESTOPTS)
