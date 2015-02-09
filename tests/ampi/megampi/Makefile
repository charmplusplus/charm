-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: pgm

pgm: test.o
	$(CHARMC) -o pgm test.o -balancer RandCentLB

test.o: test.C
	$(CHARMC) -c test.C

#
#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod pgm *~ conv-host charmrun test.o pgm.exe pgm.pdb pgm.ilk

test: pgm
	$(call run, ./pgm +p1 +vp1 )
	$(call run, ./pgm +p1 +vp2 )
	$(call run, ./pgm +p1 +vp4 )
	$(call run, ./pgm +p2 +vp2 )
	$(call run, ./pgm +p2 +vp4 )
	$(call run, ./pgm +p2 +vp1 )

bgtest: pgm
	$(call run, ./pgm +p1 +vp1 +x1 +y1 +z1 )
	$(call run, ./pgm +p1 +vp2 +x1 +y1 +z1 )
	$(call run, ./pgm +p1 +vp4 +x1 +y1 +z1 )
	$(call run, ./pgm +p2 +vp2 +x2 +y1 +z1 )
	$(call run, ./pgm +p2 +vp4 +x2 +y1 +z1 )
	$(call run, ./pgm +p2 +vp1 +x2 +y1 +z1 )
