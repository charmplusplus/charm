-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS) 

all: pgm 

pgm: pgm.o
	$(CHARMC) -o pgm pgm.o -language fem -module netfem -lmoduleifem

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

clean:
	rm -f pgm fpgm *.o conv-host charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk

test: pgm
	$(call run, ./pgm +p1 +vp1 )
	$(call run, ./pgm +p1 +vp2 )
	$(call run, ./pgm +p2 +vp1 )
	$(call run, ./pgm +p2 +vp2 )
	$(call run, ./pgm +p2 +vp3 )
	$(call run, ./pgm +p2 +vp4 )

bgtest: pgm
	$(call run, ./pgm +p1 +vp1 +x1 +y1 +z1 )
	$(call run, ./pgm +p1 +vp2 +x1 +y1 +z1 )
	$(call run, ./pgm +p2 +vp1 +x2 +y1 +z1 )
	$(call run, ./pgm +p2 +vp2 +x2 +y1 +z1 )
	$(call run, ./pgm +p2 +vp3 +x2 +y1 +z1 )
	$(call run, ./pgm +p2 +vp4 +x2 +y1 +z1 )
