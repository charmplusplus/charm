-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/ampicc

all: memckpt

ampi: memckpt

memckpt: memckpt.c
	$(CHARMC) -o memckpt memckpt.c $(OPTS)

# In-memory checkpoint at iteration 5, rollback at iteration 8 (needs the
# syncft build option; tests/ampi/Makefile lists this directory only then).
# -info passes MPI_Info objects built with MPI_Info_set instead of the
# built-in ones; -second takes a second checkpoint after the rollback and
# rolls back to it; -norollback checkpoints only. With TESTRUN_PROCS > 1 the
# runs use that many processes; a rollback restores every rank on the PE
# that checkpointed it, so nothing crosses a process.
test: memckpt
	$(call run, +p2 ./memckpt +vp4 -norollback )
	$(call run, +p1 ./memckpt +vp4 )
	$(call run, +p2 ./memckpt +vp8 -info )
	$(call run, +p4 ./memckpt +vp8 )
	$(call run, +p2 ./memckpt +vp4 -second )
	$(call run, +p4 ./memckpt +vp8 -second -info )

clean:
	rm -f *.o *.mod moduleinit.C memckpt *~ conv-host charmrun ampirun
