-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: megampi

megampi: test.o
	$(CHARMC) -o megampi test.o -balancer TreeLB

test.o: test.C
	$(CHARMC) -c test.C
#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod megampi bg_* *~ conv-host charmrun test.o megampi.exe megampi.pdb megampi.ilk ampirun

test: megampi
	$(call run, ./megampi +p1 +vp1 +balancer RandCentLB)
	$(call run, ./megampi +p1 +vp2 +balancer RandCentLB)
	$(call run, ./megampi +p1 +vp4 +balancer RandCentLB)
	$(call run, ./megampi +p2 +vp2 +balancer RandCentLB)
	$(call run, ./megampi +p2 +vp4 +balancer RandCentLB)
	$(call run, ./megampi +p2 +vp1 +balancer RandCentLB)

testp: megampi
	$(call run, ./megampi +p$(P) +vp$(P) )
	$(call run, ./megampi +p$(P) +vp$$(( $(P) * 2 )) )
	$(call run, ./megampi +p$(P) +vp$$(( $(P) * 4 )) )
