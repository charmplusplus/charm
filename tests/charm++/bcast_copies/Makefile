-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

# Not in tests/charm++/Makefile DIRS on purpose: a stress-and-audit program
# for hand runs (see README), not a CI test.

all: nokeepaudit

nokeepaudit: nokeepaudit.decl.h nokeepaudit.def.h nokeepaudit.C
	$(CHARMC) -language charm++ nokeepaudit.C -o nokeepaudit

nokeepaudit.decl.h nokeepaudit.def.h: nokeepaudit.ci
	$(CHARMC) nokeepaudit.ci

clean:
	rm -f *.decl.h *.def.h *.o nokeepaudit charmrun

# Copy counts for every entry kind, then the migration stress.
test: nokeepaudit
	$(call run, ./nokeepaudit +p4 3 1024 0 0 200)

# Also the [threaded, nokeep] cases (bit 1 and 4) and the custom-packed
# array broadcast to a keep entry (bit 2); all safe with charm PRs
# #4000/#4001 and #4002/#4003.
testall: nokeepaudit
	$(call run, ./nokeepaudit +p4 3 1024 0 7 200)

smptest: nokeepaudit
	$(call run, ./nokeepaudit +p4 ++ppn 2 3 1024 0 7 200)
