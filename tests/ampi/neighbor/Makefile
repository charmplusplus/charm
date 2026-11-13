-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: neighbor

neighbor: neighbor.o
	$(CHARMC) -o neighbor neighbor.o

neighbor.o: neighbor.C
	$(CHARMC) -c neighbor.C

test: neighbor
	$(call run, ./neighbor +p1 +vp4)
	$(call run, ./neighbor +p2 +vp4)
	$(call run, ./neighbor +p2 +vp7)

testp: neighbor
	$(call run, ./neighbor +p$(P) +vp$(P) )

clean:
	rm -f *.o *.mod neighbor *~ conv-host charmrun neighbor.o charmrun.exe neighbor.exe neighbor.pdb neighbor.ilk ampirun
