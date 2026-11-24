-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: xarray_section_reduction

xarray_section_reduction.decl.h xarray_section_reduction.def.h: xarray_section_reduction.ci
	$(CHARMC) xarray_section_reduction.ci

xarray_section_reduction: xarray_section_reduction.C xarray_section_reduction.decl.h xarray_section_reduction.def.h
	$(CHARMC) -language charm++ xarray_section_reduction.C -o xarray_section_reduction -module CkMulticast

# The program aborts on its own after 20 s without progress.
test: xarray_section_reduction
	$(call run, ./xarray_section_reduction +p1)
	$(call run, ./xarray_section_reduction +p2)
	$(call run, ./xarray_section_reduction +p4)

smptest: xarray_section_reduction

clean:
	rm -f *.decl.h *.def.h *.o xarray_section_reduction charmrun
