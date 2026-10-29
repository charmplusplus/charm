-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/ampicc

all: hello

ampi: hello

hello: hello.c
	$(CHARMC) -o hello hello.c $(OPTS)

# Checkpoint to the directory log/, then restart from it on the same, fewer
# and more PEs. With TESTRUN_PROCS > 1 the runs use that many processes, and
# the last checkpoint (written by that many processes) is also restarted in
# one process. A multi-process restart with a different process count is
# what exercises the Isomalloc region recorded in the checkpoint
# (log/Isomalloc.dat): every process must reserve the same address range
# that the checkpointed threads' stacks and heaps used.
test: hello
	rm -rf log
	$(call run, +p4 ./hello +vp8 )
	-sync
	$(call run, +p4 ./hello +vp8 +restart log )
	$(call run, +p2 ./hello +vp8 +restart log )
	$(call run, +p8 ./hello +vp8 +restart log )
	-sync
	rm -rf log
	$(call run, +p2 ./hello +vp8 )
	-sync
	$(call run, +p4 ./hello +vp8 +restart log )
ifneq ($(filter-out 0 1,$(TESTRUN_PROCS)),)
	TESTRUN_PROCS=1 $(call run, +p4 ./hello +vp8 +restart log )
endif

clean:
	rm -f *.o *.mod moduleinit.C hello *~ conv-host charmrun ampirun
	rm -rf log
