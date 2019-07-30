-include ../../common.mk
CHARMC=../../../bin/ampicxx $(OPTS)

all: pgm bg_pgm

pgm: test.o
	$(CHARMC) -o pgm test.o -balancer RandCentLB

test.o: test.C
	$(CHARMC) -c test.C

bg_pgm: bg_test.o
	$(CHARMC) -o bg_pgm bg_test.o -balancer RandCentLB

bg_test.o: test.C
	$(CHARMC) -DBIGSIM_TEST=1 -o bg_test.o -c test.C

#
#
# clean up .o, .mod, .exe and EMACS backup files
#
clean:
	rm -f *.o *.mod pgm bg_* *~ conv-host charmrun test.o pgm.exe pgm.pdb pgm.ilk ampirun

test: pgm
	$(call run, ./pgm +p1 +vp1 )
	$(call run, ./pgm +p1 +vp2 )
	$(call run, ./pgm +p1 +vp4 )
	$(call run, ./pgm +p2 +vp2 )
	$(call run, ./pgm +p2 +vp4 )
	$(call run, ./pgm +p2 +vp1 )

smptest: pgm
	$(call run, ./pgm +p2 ++ppn 2 +vp2 )
	$(call run, ./pgm +p2 ++ppn 2 +vp4 )
	$(call run, ./pgm +p2 ++ppn 2 +vp1 )
	$(call run, ./pgm +p4 ++ppn 2 +vp4 )
	$(call run, ./pgm +p4 ++ppn 2 +vp8 )
	$(call run, ./pgm +p4 ++ppn 2 +vp1 )

bgtest: bg_pgm
	$(call run, ./bg_pgm +p1 +vp1 +x1 +y1 +z1 )
	$(call run, ./bg_pgm +p1 +vp2 +x1 +y1 +z1 )
	$(call run, ./bg_pgm +p1 +vp4 +x1 +y1 +z1 )
	$(call run, ./bg_pgm +p2 +vp2 +x2 +y1 +z1 )
	$(call run, ./bg_pgm +p2 +vp4 +x2 +y1 +z1 )
	$(call run, ./bg_pgm +p2 +vp1 +x2 +y1 +z1 )
