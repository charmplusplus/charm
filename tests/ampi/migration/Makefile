-include ../../common.mk
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
	rm -f *.o *.mod pgm *~ conv-host charmrun test.o charmrun.exe pgm.exe pgm.pdb pgm.ilk ampirun

test: pgm
	$(call run, ./pgm +p1 +vp1 )
	$(call run, ./pgm +p1 +vp4 )
	$(call run, ./pgm +p1 +vp20 )
	$(call run, ./pgm +p2 +vp1 )
	$(call run, ./pgm +p2 +vp4 +mapping MAPFILE )
	$(call run, ./pgm +p2 +vp20 +mapping MAPFILE )

testp: pgm
	$(call run, ./pgm +p$(P) +vp$(P) )
	$(call run, ./pgm +p$(P) +vp$$(( $(P) * 2 )) )
	$(call run, ./pgm +p$(P) +vp$$(( $(P) * 4 )) )
