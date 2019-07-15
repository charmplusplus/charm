-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: sdag_tests

sdag_tests: sdag_tests.C sdag_tests.decl.h basic.decl.h refnum.decl.h migration.decl.h
	$(CHARMC) sdag_tests.C -o sdag_tests

%.decl.h: %.ci
	$(CHARMC) $<

clean:
	rm -f *.decl.h *.def.h *.o sdag_tests charmrun

test: all
	$(call run, ./sdag_tests +p1)
	$(call run, ./sdag_tests +p4)

smptest: all
	$(call run, ./sdag_tests +p2 ++ppn 2)
	$(call run, ./sdag_tests +p4 ++ppn 2)
