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
test: memckpt testrefusal
	$(call run, +p2 ./memckpt +vp4 -norollback )
	$(call run, +p1 ./memckpt +vp4 )
	$(call run, +p2 ./memckpt +vp8 -info )
	$(call run, +p4 ./memckpt +vp8 )
	$(call run, +p2 ./memckpt +vp4 -second )
	$(call run, +p4 ./memckpt +vp8 -second -info )

# A communicator created between the checkpoint and the rollback (its ranks
# are a new array bound to MPI_COMM_WORLD's) must make the runtime refuse the
# rollback. testrun skips (exit 0, "Skipping the test") when the process count
# does not divide the PE count; that counts as a pass.
# $(1): testrun and program options, $(2): the word in the refusal message
define refusal
	@$(call run, $(1) ) > refusal.out 2>&1; rc=$$?; \
	if grep -q "Skipping the test" refusal.out; then \
	  echo "memckpt testrefusal ($(1)) skipped"; \
	elif [ $$rc -eq 0 ]; then \
	  cat refusal.out; echo "testrefusal ($(1)): expected a nonzero exit"; exit 1; \
	elif ! grep -q "rollback refused: .* array elements $(2)" refusal.out; then \
	  cat refusal.out; echo "testrefusal ($(1)): refusal message missing"; exit 1; \
	else \
	  echo "memckpt testrefusal ($(1)) passed"; \
	fi
endef

testrefusal: memckpt
	$(call refusal,+p2 ./memckpt +vp4 -newcomm,inserted)

clean:
	rm -f *.o *.mod moduleinit.C memckpt *~ conv-host charmrun ampirun refusal.out
