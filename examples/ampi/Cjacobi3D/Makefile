-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMBASE=../../../
CHARMC=../../../bin/ampicxx $(OPTS)
TOKENS=6

-include $(CHARMBASE)/include/conv-mach-opt.mak

all: ampi

AMPI_TARGETS := \
  jacobi \
  jacobi.pup \
  jacobi-get \

ifeq (1,$(CMK_SUPPORTS_TLSGLOBALS))
  AMPI_TARGETS += jacobi.tls
endif

ifeq (1,$(CMK_SUPPORTS_MEMORY_ISOMALLOC))
  AMPI_TARGETS += jacobi.iso
endif

ampi: $(AMPI_TARGETS)

mpi: jacobi.C
	mpiCC -o jacobi jacobi.C  $(OPTS)

jacobi-cpp: jacobi-cpp.C
	$(CHARMC) -c jacobi-cpp.C 
	$(CHARMC) -o $@ jacobi-cpp.o -lm

jacobi.o: jacobi.C
	$(CHARMC) -c jacobi.C 

jacobi: jacobi.o
	$(CHARMC) -o $@ $^ -lm

jacobi.pup: jacobi.o
	$(CHARMC) -o $@ $^ -memory default -lm

jacobi.iso: jacobi.C
	$(CHARMC) -c -DNO_PUP jacobi.C -o jacobi.iso.o
	$(CHARMC) -o $@ jacobi.iso.o -memory isomalloc

jacobi.tls: jacobi.C
	$(CHARMC) -c -tlsglobals jacobi.C -o jacobi.tls.o
	$(CHARMC) -o jacobi.tls jacobi.tls.o -tlsglobals

jacobi.rose: jacobi.C
	$(CHARMC) -roseomptlsglobals -o jacobi.rose.o -c $<
	$(CHARMC) -roseomptlsglobals -o $@ jacobi.rose.o

# jacobi-get creates an MPI_Win over the chunk, and a window's memory has to
# keep its address when the rank migrates or restarts. The Isomalloc heap
# moves the chunk at the same address; chunk_pup, which frees the chunk and
# allocates a new one, would not. So the pup routine is left out (-DNO_PUP).
jacobi-get: jacobi-get.C
	$(CHARMC) -c -DNO_PUP jacobi-get.C
	$(CHARMC) -o $@ jacobi-get.o -lm

jacobi-get-tokens:
	cd $(CHARMBASE)/tmp; touch machine.c; touch ckfutures.C; make AMPI OPTS="-DIGET_FLOWCONTROL=1 -DIGET_TOKENNUM=$(TOKENS)"; cd -; 
	make jacobi-get;

# A restart from a checkpoint resumes each rank's thread in a new process,
# which must load the program and its libraries at the addresses the thread's
# stack holds, so address space layout randomization must be off. On Linux
# testrun runs every program under setarch -R, unless TESTRUN_VA_RANDOMIZATION=1
# asks it not to; macOS has no way to turn it off, and Isomalloc refuses the
# restart. The same condition as in tests/ampi/Makefile.
ifeq ($(CMK_GDIR),reconverse)
ifeq ($(CMK_MACOSX),1)
RECONVERSE_VA_RANDOMIZED = 1
endif
ifeq ($(TESTRUN_VA_RANDOMIZATION),1)
RECONVERSE_VA_RANDOMIZED = 1
endif
# With randomization on and more than one process, ranks must not migrate
# between processes either (Isomalloc refuses that too), and the balancer in
# the checkpoint runs would move them, so those runs are left out as well.
ifeq ($(RECONVERSE_VA_RANDOMIZED),1)
ifneq ($(filter-out 0 1,$(TESTRUN_PROCS)),)
NO_CHKPT_RUNS = 1
endif
endif
endif

# The final line of a restarted run must be the final line of the
# uninterrupted run that wrote the checkpoint. testrun skips a run (exit 0,
# "Skipping the test") when the process count does not divide the PE count;
# a skipped run, or a skipped reference run, counts as a pass.
define check_restart
	if grep -q "Skipping the test" $(1) $(2); then echo "$(1): skipped"; \
	else grep "^jacobi: " $(1) && test "$$(grep '^jacobi: ' $(1))" = "$$(grep '^jacobi: ' $(2))"; fi
endef

# A checkpoint run must complete all 40 iterations, unless testrun skipped it.
# $(1): its log
define check_chkpt
	grep -q "Skipping the test" $(1) || grep "^jacobi: 40 iterations" $(1)
endef

# Restart on $(1) PEs of program $(2) from checkpoint directory $(3), written
# by the run logged in $(4); the restart is logged in $(5). When testrun
# skipped the checkpoint run there is nothing to restart from, so the
# restart is skipped too.
define restart
	if grep -q "Skipping the test" $(4); then echo "Skipping the test: $(4) wrote no checkpoint" > $(5); \
	else $(call run, +p$(1) ./$(2) 2 2 2 40 -chkpt 20 $(3) +vp8 +balancer RotateLB +restart $(3) ) 2>&1 | tee $(5); fi
	$(call check_restart,$(5),$(4))
endef

test: ampi
	$(call run, +p3 ./jacobi.pup 2 2 2 40 +vp8 +balancer RotateLB +LBDebug 1 )
	$(call run, +p3 ./jacobi 2 2 2 40 +vp8 +balancer RotateLB +LBDebug 1 )
	$(call run, +p3 ./jacobi-get 2 2 2 40 +vp8 +balancer RotateLB +LBDebug 1 )
ifeq (1,$(CMK_SUPPORTS_TLSGLOBALS))
	$(call run, +p3 ./jacobi.tls 2 2 2 40 +vp8 +balancer RotateLB +LBDebug 1)
endif
ifeq (1,$(CMK_SUPPORTS_MEMORY_ISOMALLOC))
	$(call run, +p3 ./jacobi.iso 2 2 2 40 +vp8 +balancer RotateLB +LBDebug 1)
endif
	# Checkpoint after iteration 20 of 40 to log/ (the load balancer keeps
	# moving ranks at iterations 5, 15, 25, 35), then restart from it on the
	# same, fewer and more PEs. With TESTRUN_PROCS set, testrun skips a PE
	# count the process count does not divide, and the checks above accept
	# the skip. The Isomalloc build (jacobi) restores the stack and heap, the
	# -memory default build (jacobi.pup) restores the chunk with chunk_pup.
ifneq ($(NO_CHKPT_RUNS),1)
	rm -rf log log.pup
	$(call run, +p4 ./jacobi 2 2 2 40 -chkpt 20 log +vp8 +balancer RotateLB ) 2>&1 | tee chkpt.log
	$(call check_chkpt,chkpt.log)
	$(call run, +p4 ./jacobi.pup 2 2 2 40 -chkpt 20 log.pup +vp8 +balancer RotateLB ) 2>&1 | tee chkpt.pup.log
	$(call check_chkpt,chkpt.pup.log)
ifneq ($(RECONVERSE_VA_RANDOMIZED),1)
	-sync
	$(call restart,4,jacobi,log,chkpt.log,restart4.log)
	$(call restart,2,jacobi,log,chkpt.log,restart2.log)
	$(call restart,6,jacobi,log,chkpt.log,restart6.log)
	$(call restart,6,jacobi.pup,log.pup,chkpt.pup.log,restart6.pup.log)
endif
endif

# In-memory checkpoint after iteration 12 of 30 and rollback after 18 (needs
# the syncft build option): the final line must be that of a run without
# them. Load balancing runs at iteration 5, before the checkpoint, and at 25,
# after the rollback; it is skipped at 15. The runs with RotateLB move ranks,
# so they are left out when ranks cannot migrate between processes (above).
testmemckpt: ampi
ifeq ($(CMK_MEM_CHECKPOINT),1)
	$(call run, +p4 ./jacobi 2 2 2 30 +vp8 ) 2>&1 | tee memckpt-ref.log
	$(call run, +p4 ./jacobi 2 2 2 30 -memckpt 12 -rollback 18 +vp8 ) 2>&1 | tee memckpt.log
	grep "rolling back to iteration 12" memckpt.log
	grep "after the in-memory checkpoint, rollbacks 1" memckpt.log
	$(call check_restart,memckpt.log,memckpt-ref.log)
	$(call run, +p2 ./jacobi.pup 2 2 2 30 -memckpt 12 -rollback 18 +vp8 ) 2>&1 | tee memckpt.pup.log
	$(call check_restart,memckpt.pup.log,memckpt-ref.log)
ifneq ($(NO_CHKPT_RUNS),1)
	$(call run, +p4 ./jacobi 2 2 2 30 -memckpt 12 -rollback 18 +vp8 +balancer RotateLB ) 2>&1 | tee memckpt.lb.log
	$(call check_restart,memckpt.lb.log,memckpt-ref.log)
endif
else
	@echo "testmemckpt: needs a build with the syncft option; skipped"
endif

testp: ampi
	$(call run, +p$(P) ./jacobi.pup $(P) 1 1 40 +vp$(P) +balancer RotateLB +LBDebug 1 )
	$(call run, +p$(P) ./jacobi.pup $(P) 2 1 40 +vp$$(( $(P) * 2 )) +balancer RotateLB +LBDebug 1 )
	$(call run, +p$(P) ./jacobi.pup $(P) 2 2 40 +vp$$(( $(P) * 4 )) +balancer RotateLB +LBDebug 1 )

	$(call run, +p$(P) ./jacobi $(P) 1 1 40 +vp$(P) +balancer RotateLB +LBDebug 1 )
	$(call run, +p$(P) ./jacobi $(P) 2 1 40 +vp$$(( $(P) * 2 )) +balancer RotateLB +LBDebug 1 )
	$(call run, +p$(P) ./jacobi $(P) 2 2 40 +vp$$(( $(P) * 4 )) +balancer RotateLB +LBDebug 1 )

ifeq (1,$(CMK_SUPPORTS_TLSGLOBALS))
	$(call run, +p$(P) ./jacobi.tls $(P) 1 1 40 +vp$(P) +balancer RotateLB +LBDebug 1 )
	$(call run, +p$(P) ./jacobi.tls $(P) 2 1 40 +vp$$(( $(P) * 2 )) +balancer RotateLB +LBDebug 1 )
	$(call run, +p$(P) ./jacobi.tls $(P) 2 2 40 +vp$$(( $(P) * 4 )) +balancer RotateLB +LBDebug 1 )
endif

ifeq (1,$(CMK_SUPPORTS_MEMORY_ISOMALLOC))
	$(call run, +p$(P) ./jacobi.iso $(P) 1 1 40 +vp$(P) +balancer RotateLB +LBDebug 1)
	$(call run, +p$(P) ./jacobi.iso $(P) 2 1 40 +vp$$(( $(P) * 2 )) +balancer RotateLB +LBDebug 1 )
	$(call run, +p$(P) ./jacobi.iso $(P) 2 2 40 +vp$$(( $(P) * 4 )) +balancer RotateLB +LBDebug 1 )
endif



clean:
	rm -f *.o jacobi jacobi.pup *~ moduleinit.C charmrun conv-host jacobi-cpp jacobi.iso jacobi-get jacobi.tls ampirun
	rm -f chkpt.log chkpt.pup.log restart*.log memckpt*.log
	rm -rf 40 80 120 log log.pup
