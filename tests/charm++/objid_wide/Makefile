-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: objid_wide

# GreedyLB comes from CommonLBs.
objid_wide: objid_wide.decl.h objid_wide.C
	$(CHARMC) objid_wide.C -o objid_wide -module CommonLBs

objid_wide.decl.h: objid_wide.ci
	$(CHARMC) objid_wide.ci

# The abort runs exit non-zero, which the pipe hides; grep checks the text and
# reads to the end (no -q: closing the pipe early kills the program with SIGPIPE).
# The abort cases run through testrun but not under TESTRUN_PROCS: a process that
# aborts does not end a multi-process lcrun job, the other process spins forever.
test: objid_wide
	$(call run, ./objid_wide +p4 )
	$(call run, ./objid_wide +p4 -wide 2000 -lb +balancer GreedyLB )
	-rm -rf ckptlog
	$(call run, ./objid_wide +p4 -wide 3000 -ckpt )
	$(call run, ./objid_wide +p4 +restart ckptlog )
	$(call run, ./objid_wide +p2 +restart ckptlog )
	-rm -rf ckptlog
	$(call run, ./objid_wide +p4 -wide 1000 -bound )
	$(call run, ./objid_wide +p4 -wide 1000 -bound -lb +balancer GreedyLB )
	$(call run, ./objid_wide +p4 -wide 1000 -bound -ckpt )
	$(call run, ./objid_wide +p4 +restart ckptlog )
	$(call run, ./objid_wide +p2 +restart ckptlog )
	-rm -rf ckptlog
	$(call run, ./objid_wide +p4 -wide 10 -standard 2100) 2>&1 | grep "chare arrays created, half of the" > /dev/null
ifeq ($(TESTRUN_PROCS),)
	$(call run, ./objid_wide +p4 -abort nobounds) 2>&1 | grep "but no bounds" > /dev/null
	$(call run, ./objid_wide +p4 -abort bigbounds) 2>&1 | grep "bits to pack its index, but a wide-collection" > /dev/null
	$(call run, ./objid_wide +p4 -wide 10 -standard 4200) 2>&1 | grep "ran out of collection numbers" > /dev/null
else
	@echo "objid_wide: the three abort cases are skipped under TESTRUN_PROCS (an abort does not end a multi-process lcrun job)"
endif

smptest: test

clean:
	rm -f *.decl.h *.def.h *.o objid_wide charmrun
	rm -rf ckptlog/
