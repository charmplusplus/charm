-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: boundmany

boundmany: boundmany.decl.h boundmany.C
	$(CHARMC) boundmany.C -o boundmany

boundmany.decl.h: boundmany.ci
	$(CHARMC) boundmany.ci

# The bound arrays' creations reach the PEs in different orders under the
# randomized queue. Without the fix for migration by array id, 8 PEs and the
# default 64 arrays failed every seed.
test: boundmany
	$(call run, ./boundmany +p8)
	$(call run, ./boundmany +p8 +randomized_msgq +randomized_seed 1)
	$(call run, ./boundmany +p8 +randomized_msgq +randomized_seed 2)
	$(call run, ./boundmany +p8 +randomized_msgq +randomized_seed 3)
	$(call run, ./boundmany +p8 +randomized_msgq +randomized_seed 4)

	# Checkpoint before round 3, restart on fewer and on the same PEs. Without
	# the array ids in the checkpoint, restarts on 4 PEs failed every seed.
	-rm -rf log
	$(call run, ./boundmany +p8 -ckpt-at 3 +randomized_msgq +randomized_seed 1)
	-sync
	$(call run, ./boundmany +p4 +restart log)
	$(call run, ./boundmany +p8 +restart log)
	-rm -rf log

smptest: boundmany
	$(call run, ./boundmany +p8 +randomized_msgq +randomized_seed 1)

clean:
	rm -rf *.decl.h *.def.h *.o boundmany charmrun log
