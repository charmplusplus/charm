-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: ckcoro_tests names.o

ckcoro_tests: ckcoro_tests.C ckcoro_tests.decl.h basic.decl.h refnum.decl.h migration.decl.h owners.decl.h ../../../include/ckcoro.h
	$(CHARMC) -std=c++20 ckcoro_tests.C -o ckcoro_tests

# The two attribute words used as names (names.ci): compiled, not run.
names.o: names.C names.decl.h
	$(CHARMC) -c names.C -o names.o

%.decl.h: %.ci
	$(CHARMC) $<

clean:
	rm -f *.decl.h *.def.h *.o ckcoro_tests charmrun
	rm -rf log

test: all
	$(call run, ./ckcoro_tests +p1)
	$(call run, ./ckcoro_tests +p4)
	$(call run, ./ckcoro_tests +p4 +ck_fakemigrate)
	rm -rf log
	$(call run, ./ckcoro_tests -chkpt log +p4)
	-sync
	$(call run, ./ckcoro_tests +p4 +restart log)

.PHONY: all clean test
