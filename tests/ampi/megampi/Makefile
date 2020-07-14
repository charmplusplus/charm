-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: pgm

pgm: test.o
	$(CHARMC) -o pgm test.o -balancer TreeLB

test.o: test.C
	$(CHARMC) -c test.C
#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod pgm bg_* *~ conv-host charmrun test.o pgm.exe pgm.pdb pgm.ilk ampirun

test: pgm
	$(call run, ./pgm +p1 +vp1 +balancer RandCentLB)
	$(call run, ./pgm +p1 +vp2 +balancer RandCentLB)
	$(call run, ./pgm +p1 +vp4 +balancer RandCentLB)
	$(call run, ./pgm +p2 +vp2 +balancer RandCentLB)
	$(call run, ./pgm +p2 +vp4 +balancer RandCentLB)
	$(call run, ./pgm +p2 +vp1 +balancer RandCentLB)

testp: pgm
	$(call run, ./pgm +p$(P) +vp1 )
	$(call run, ./pgm +p$(P) +vp2 )
	$(call run, ./pgm +p$(P) +vp4 )
