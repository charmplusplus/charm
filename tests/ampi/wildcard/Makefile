-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: wildcard

wildcard: wildcard.o
	$(CHARMC) -o wildcard wildcard.o

wildcard.o: wildcard.C
	$(CHARMC) -c wildcard.C

test: wildcard
	$(call run, ./wildcard +p1 +vp4)
	$(call run, ./wildcard +p2 +vp4)

testp: wildcard
	$(call run, ./wildcard +p$(P) +vp$(P) )
	$(call run, ./wildcard +p$(P) +vp$$(( $(P) * 2 )) )

clean:
	rm -f *.o *.mod wildcard *~ conv-host charmrun ampirun wildcard.exe wildcard.pdb wildcard.ilk
