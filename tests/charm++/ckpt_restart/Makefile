-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: ckpt_restart
	rm -rf log/

# GreedyLB comes from CommonLBs; RotateLB is built into the load balancing manager.
ckpt_restart: ckpt_restart.decl.h ckpt_restart.C
	$(CHARMC) -language charm++ ckpt_restart.C -o ckpt_restart -module CommonLBs

ckpt_restart.decl.h: ckpt_restart.ci
	$(CHARMC) ckpt_restart.ci

# Windows CI sets RUN_PROCS= (no multi-process runs there), which makes the
# two-process lines single-process ones.
RUN_PROCS = TESTRUN_PROCS=2

RUN = ./ckpt_restart -steps 16
CKPT = $(RUN) -ckpt-at 6

# Every checkpoint run checkpoints to log/ after step 6 (one load balancing step,
# at step 4, has moved the ring elements) and every restart must end with the
# summary of the uninterrupted reference run (ref.chk).
#
# CI runs make test twice, plain and with TESTRUN_PROCS=2. The lines that set
# their own process count (TESTRUN_PROCS=1 or 2 in front of the run) are the
# point of this test, the checkpoint written by one number of processes and
# restarted in another; they run in the plain pass only, as running them again
# with TESTRUN_PROCS=2 would repeat the same runs (and the plain pass is the one
# a local make test runs). The other lines run in both
# passes: one process in the plain pass, two processes with TESTRUN_PROCS=2.
test: all
	-rm -rf log ref.chk
	# Reference: no checkpoint, at 1 PE.
	$(call run, $(RUN) +p1 -write-check ref.chk )
ifeq ($(filter-out 0 1,$(TESTRUN_PROCS)),)
	# Checkpoint in 2 processes, restart in 1 process on fewer and the same PEs.
	$(RUN_PROCS) $(call run, $(CKPT) +p4 +balancer RotateLB -expect-check ref.chk )
	-sync
	TESTRUN_PROCS=1 $(call run, ./ckpt_restart +p3 +restart log -expect-check ref.chk )
	TESTRUN_PROCS=1 $(call run, ./ckpt_restart +p4 +restart log -expect-check ref.chk )
	-rm -rf log
	# Checkpoint in 1 process, restart in 2 processes on more PEs.
	TESTRUN_PROCS=1 $(call run, $(CKPT) +p2 +balancer RotateLB -expect-check ref.chk )
	-sync
	$(RUN_PROCS) $(call run, ./ckpt_restart +p4 +restart log -expect-check ref.chk )
	$(RUN_PROCS) $(call run, ./ckpt_restart +p6 +restart log -expect-check ref.chk )
	-rm -rf log
endif
	# The same process count, with GreedyLB: restart on the same and on fewer PEs.
	$(call run, $(CKPT) +p4 +balancer GreedyLB +LBDebug 1 -expect-check ref.chk )
	-sync
	$(call run, ./ckpt_restart +p4 +restart log +LBDebug 1 -expect-check ref.chk )
	$(call run, ./ckpt_restart +p2 +restart log +LBDebug 1 -expect-check ref.chk )
	-rm -rf log
	# No balancer; restart on more PEs.
	$(call run, $(CKPT) +p2 -expect-check ref.chk )
	-sync
	$(call run, ./ckpt_restart +p4 +restart log -expect-check ref.chk )
	$(call run, ./ckpt_restart +p1 +restart log -expect-check ref.chk )
	-rm -rf log
	# A chain of three restarts, each from the checkpoint the previous one wrote (-c).
	$(call run, $(CKPT) +p4 +balancer RotateLB -expect-check ref.chk )
	$(call run, ./ckpt_restart +p4 +restart log -c -expect-check ref.chk )
	$(call run, ./ckpt_restart +p4 +restart log -c -expect-check ref.chk )
	$(call run, ./ckpt_restart +p4 +restart log -expect-check ref.chk )
	-rm -rf log ref.chk

# Checkpoint on $(P) PEs, restart on $(P) and $(P)/2 PEs; then on $(P)/2, restart on $(P).
testp: all
	-rm -rf log ref.chk
	$(call run, $(RUN) +p1 -write-check ref.chk )
	$(call run, $(CKPT) +p$(P) +balancer RotateLB -expect-check ref.chk )
	-sync
	$(call run, ./ckpt_restart +p$(P) +restart log -expect-check ref.chk )
	$(call run, ./ckpt_restart +p$$(( $(P) / 2 )) +restart log -expect-check ref.chk )
	-rm -rf log
	$(call run, $(CKPT) +p$$(( $(P) / 2 )) -expect-check ref.chk )
	-sync
	$(call run, ./ckpt_restart +p$(P) +restart log -expect-check ref.chk )
	-rm -rf log ref.chk

# A chain of nine restarts. Before 2026-10-10 every restart spent 1/8 of the
# allocator's pool of unique numbers and the ninth aborted ("has used all 2^U
# unique element ids"); the "Generation N: the previous restart consumed" lines
# show what each restart costs now.
testchain: all
	-rm -rf log ref.chk
	$(call run, $(RUN) +p1 -write-check ref.chk )
	$(call run, $(CKPT) +p4 -expect-check ref.chk )
	for i in 1 2 3 4 5 6 7 8; do \
	  $(call run, ./ckpt_restart +p4 +restart log -c -expect-check ref.chk ) || exit 1; \
	done
	$(call run, ./ckpt_restart +p4 +restart log -expect-check ref.chk )
	-rm -rf log ref.chk

# A chain under a narrow pool: +objid_expand 2^24 leaves U = 25 unique bits, so
# the allocator's cap on 1 process is 2^21 and the first grant after a restart
# is cap / 64 = 2^15, not the flat 2^16 (see the "Generation N" lines).
testnarrow: all
	-rm -rf log ref.chk
	$(call run, $(RUN) +p1 -write-check ref.chk )
	$(call run, $(CKPT) +p4 +objid_expand 16777216 -expect-check ref.chk )
	$(call run, ./ckpt_restart +p4 +restart log -c -expect-check ref.chk )
	$(call run, ./ckpt_restart +p4 +restart log -c -expect-check ref.chk )
	$(call run, ./ckpt_restart +p4 +restart log -c -expect-check ref.chk )
	$(call run, ./ckpt_restart +p2 +restart log -c -expect-check ref.chk )
	-rm -rf log ref.chk

# tests/charm++/Makefile runs smptest in every directory on SMP builds. On
# reconverse testrun drops ++ppn, so an SMP variant of the runs above would only
# repeat them; the process-count changes are what test runs instead.
smptest:
	@echo "ckpt_restart: no separate smptest (make test covers 1 and 2 processes)"

clean:
	rm -f *.decl.h *.def.h *.o ckpt_restart charmrun ref.chk
	rm -rf log/
