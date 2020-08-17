-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS) 
OBJS=pgm.o ctests.o 
FOBJS=$(OBJS) ftests.o
COBJS=$(OBJS) ftests_fallback.o

TARGETS := pgm
ifeq (1,$(CMK_CAN_LINK_FORTRAN))
  TARGETS += fpgm
endif

all: $(TARGETS)

pgm: $(COBJS)
	$(CHARMC) -o $@ $^ -language ampi -module fem

fpgm: $(FOBJS)
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
	rm -f pgm fpgm *.o *.MOD *.mod charmrun

test: $(addprefix test-,$(TARGETS))

test-pgm: pgm
	$(call run, ./pgm +p1 +vp1)
	$(call run, ./pgm +p1 +vp2)
	$(call run, ./pgm +p2 +vp1)
	$(call run, ./pgm +p2 +vp2)
	$(call run, ./pgm +p2 +vp3)
	$(call run, ./pgm +p2 +vp4)
	-@rm -f fem_mesh_vp3_*
	$(call run, ./pgm +p1 +vp3 -write)
	$(call run, ./pgm +p1 +vp3 -read)
	rm -f fem_mesh_vp3_*

test-fpgm: fpgm
	$(call run, ./fpgm +p1 +vp1)
	$(call run, ./fpgm +p1 +vp2)
	$(call run, ./fpgm +p2 +vp1)
	$(call run, ./fpgm +p2 +vp2)
	$(call run, ./fpgm +p2 +vp3)
	$(call run, ./fpgm +p2 +vp4)
	-@rm -f fem_mesh_vp3_*
	$(call run, ./fpgm +p1 +vp3 -write)
	$(call run, ./fpgm +p1 +vp3 -read)
	rm -f fem_mesh_vp3_*

.PHONY: all clean test test-pgm test-fpgm
