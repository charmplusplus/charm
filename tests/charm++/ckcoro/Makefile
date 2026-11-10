-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: ckcoro_tests

ckcoro_tests: ckcoro_tests.C ckcoro_tests.decl.h migration.decl.h ../../../include/ckcoro.h
	$(CHARMC) -std=c++20 ckcoro_tests.C -o ckcoro_tests

%.decl.h: %.ci
	$(CHARMC) $<

clean:
	rm -f *.decl.h *.def.h *.o ckcoro_tests charmrun

test: all
	$(call run, ./ckcoro_tests +p1)
	$(call run, ./ckcoro_tests +p4)
	$(call run, ./ckcoro_tests +p4 +ck_fakemigrate)

.PHONY: all clean test
