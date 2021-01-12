-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

TARGETS := femtest
ifeq (1,$(CMK_SUPPORTS_MEMORY_ISOMALLOC))
  TARGETS += femtest-mig
endif
ifeq (1,$(CMK_CAN_LINK_FORTRAN))
  TARGETS += ffemtest
endif

all: $(TARGETS)

femtest: pgm.o
	$(CHARMC) -o $@ $^ -language fem

pgm.o: pgm.C
	$(CHARMC) -o $@ -c $<

femtest-mig: pgm-mig.o
	$(CHARMC) -o $@ $^ -language fem -DENABLE_MIG=1 -memory isomalloc -module TreeLB

pgm-mig.o: pgm.C
	$(CHARMC) -o $@ -c $< -DENABLE_MIG=1 -memory isomalloc -module TreeLB

ffemtest: fpgm.o
	$(CHARMC) -o $@ $^ -language femf

fpgm.o: fpgm.f90
	$(CHARMC) -o $@ -c $<

clean:
	rm -f femtest femtest-mig ffemtest *.o conv-host charmrun charmrun.exe femtest.exe femtest.pdb femtest.ilk FEMVERSION

test: $(addprefix test-,$(TARGETS))

test-femtest: femtest
	$(call run, ./femtest +p1 +vp1 )
	$(call run, ./femtest +p1 +vp2 )
	$(call run, ./femtest +p2 +vp1 )
	$(call run, ./femtest +p2 +vp2 )
	$(call run, ./femtest +p2 +vp3 )
	$(call run, ./femtest +p2 +vp4 )
	-@rm -f fem_mesh_vp3_*
	$(call run, ./femtest +p1 +vp3 -write )
	$(call run, ./femtest +p1 +vp3 -read )
	rm -f fem_mesh_vp3_*

test-femtest-mig: femtest-mig
	$(call run, ./femtest-mig +p2 +vp2 +balancer RandCentLB )
	$(call run, ./femtest-mig +p2 +vp3 +balancer RandCentLB )
	$(call run, ./femtest-mig +p2 +vp4 +balancer RandCentLB )

test-ffemtest: ffemtest
	$(call run, ./ffemtest +p1 +vp1 )
	$(call run, ./ffemtest +p1 +vp2 )
	$(call run, ./ffemtest +p2 +vp1 )
	$(call run, ./ffemtest +p2 +vp2 )
	$(call run, ./ffemtest +p2 +vp3 )
	$(call run, ./ffemtest +p2 +vp4 )
	-@rm -f fem_mesh_vp3_*
	$(call run, ./ffemtest +p1 +vp3 -write )
	$(call run, ./ffemtest +p1 +vp3 -read )
	rm -f fem_mesh_vp3_*

testf: test-ffemtest
test-mig: test-femtest-mig

.PHONY: all clean test test-femtest test-femtest-mig test-ffemtest testf test-mig
