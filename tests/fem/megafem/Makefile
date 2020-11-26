-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)
OBJS=pgm.o ctests.o
FOBJS=$(OBJS) ftests.o
COBJS=$(OBJS) ftests_fallback.o

TARGETS := megafem
ifeq (1,$(CMK_CAN_LINK_FORTRAN))
  TARGETS += fmegafem
endif

all: $(TARGETS)

megafem: $(COBJS)
	$(CHARMC) -o $@ $^ -language ampi -module fem

fmegafem: $(FOBJS)
	$(CHARMC) -o $@ $^ -language ampif -module fem

pgm.o: pgm.C
	$(CHARMC) -c pgm.C

ctests.o: ctests.C *.tst
	$(CHARMC) -c ctests.C

ftests.o: ftests.F90 *.tst
	$(CHARMC) -c ftests.F90 -o $@

ftests_fallback.o: ftests_fallback.C
	$(CHARMC) -c ftests_fallback.C -o $@

clean:
	rm -f megafem fmegafem *.o *.MOD *.mod charmrun

test: $(addprefix test-,$(TARGETS))

test-megafem: megafem
	$(call run, ./megafem +p1 +vp1)
	$(call run, ./megafem +p1 +vp2)
	$(call run, ./megafem +p2 +vp1)
	$(call run, ./megafem +p2 +vp2)
	$(call run, ./megafem +p2 +vp3)
	$(call run, ./megafem +p2 +vp4)
	-@rm -f fem_mesh_vp3_*
	$(call run, ./megafem +p1 +vp3 -write)
	$(call run, ./megafem +p1 +vp3 -read)
	rm -f fem_mesh_vp3_*

test-fmegafem: fmegafem
	$(call run, ./fmegafem +p1 +vp1)
	$(call run, ./fmegafem +p1 +vp2)
	$(call run, ./fmegafem +p2 +vp1)
	$(call run, ./fmegafem +p2 +vp2)
	$(call run, ./fmegafem +p2 +vp3)
	$(call run, ./fmegafem +p2 +vp4)
	-@rm -f fem_mesh_vp3_*
	$(call run, ./fmegafem +p1 +vp3 -write)
	$(call run, ./fmegafem +p1 +vp3 -read)
	rm -f fem_mesh_vp3_*

.PHONY: all clean test test-megafem test-fmegafem
