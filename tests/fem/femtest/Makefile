-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS) 

TARGETS := pgm
ifeq (1,$(CMK_SUPPORTS_MEMORY_ISOMALLOC))
  TARGETS += pgm-mig
endif
ifeq (1,$(CMK_CAN_LINK_FORTRAN))
  TARGETS += fpgm
endif

all: $(TARGETS)

pgm: pgm.o
	$(CHARMC) -o $@ $^ -language fem

pgm.o: pgm.C
	$(CHARMC) -o $@ -c $<

pgm-mig: pgm-mig.o
	$(CHARMC) -o $@ $^ -language fem -DENABLE_MIG=1 -memory isomalloc -module TreeLB

pgm-mig.o: pgm.C
	$(CHARMC) -o $@ -c $< -DENABLE_MIG=1 -memory isomalloc -module TreeLB

fpgm: fpgm.o
	$(CHARMC) -o $@ $^ -language femf

fpgm.o: fpgm.f90
	$(CHARMC) -o $@ -c $<

clean:
	rm -f pgm pgm-mig fpgm *.o conv-host charmrun charmrun.exe pgm.exe pgm.pdb pgm.ilk FEMVERSION

test: $(addprefix test-,$(TARGETS))

test-pgm: pgm
	$(call run, ./pgm +p1 +vp1 )
	$(call run, ./pgm +p1 +vp2 )
	$(call run, ./pgm +p2 +vp1 )
	$(call run, ./pgm +p2 +vp2 )
	$(call run, ./pgm +p2 +vp3 )
	$(call run, ./pgm +p2 +vp4 )
	-@rm -f fem_mesh_vp3_*
	$(call run, ./pgm +p1 +vp3 -write )
	$(call run, ./pgm +p1 +vp3 -read )
	rm -f fem_mesh_vp3_*

test-pgm-mig: pgm-mig
	$(call run, ./pgm-mig +p2 +vp2 +balancer RandCentLB )
	$(call run, ./pgm-mig +p2 +vp3 +balancer RandCentLB )
	$(call run, ./pgm-mig +p2 +vp4 +balancer RandCentLB )

test-fpgm: fpgm
	$(call run, ./fpgm +p1 +vp1 )
	$(call run, ./fpgm +p1 +vp2 )
	$(call run, ./fpgm +p2 +vp1 )
	$(call run, ./fpgm +p2 +vp2 )
	$(call run, ./fpgm +p2 +vp3 )
	$(call run, ./fpgm +p2 +vp4 )
	-@rm -f fem_mesh_vp3_*
	$(call run, ./fpgm +p1 +vp3 -write )
	$(call run, ./fpgm +p1 +vp3 -read )
	rm -f fem_mesh_vp3_*

testf: test-fpgm
test-mig: test-pgm-mig

.PHONY: all clean test test-pgm test-pgm-mig test-fpgm testf test-mig
