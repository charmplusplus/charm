-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: groupfree

groupfree: groupfree.o
	$(CHARMC) -o groupfree groupfree.o

groupfree.o: groupfree.C
	$(CHARMC) -c groupfree.C

test: groupfree
	$(call run, ./groupfree +p1 +vp2)
	$(call run, ./groupfree +p1 +vp4)
	$(call run, ./groupfree +p2 +vp4)
	$(call run, ./groupfree +p2 +vp5)

testp: groupfree
	$(call run, ./groupfree +p$(P) +vp$(P) )
	$(call run, ./groupfree +p$(P) +vp$$(( $(P) * 2 )) )

clean:
	rm -f *.o *.mod groupfree *~ conv-host charmrun groupfree.o charmrun.exe groupfree.exe groupfree.pdb groupfree.ilk ampirun
