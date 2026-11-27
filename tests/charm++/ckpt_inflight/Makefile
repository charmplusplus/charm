-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: ckpt_inflight

ckpt_inflight: ckpt_inflight.decl.h ckpt_inflight.C
	$(CHARMC) ckpt_inflight.C -o ckpt_inflight

ckpt_inflight.decl.h: ckpt_inflight.ci
	$(CHARMC) ckpt_inflight.ci

# Checkpoint while every element is migrating, then restart, for three seeds;
# then once more after every element has moved by a direct ckEmigrate.
test: ckpt_inflight
	for s in 1 2 3; do \
	  rm -rf log; \
	  $(call run, ./ckpt_inflight +p8 +randomized_msgq +randomized_seed $$s ) || exit 1; \
	  sync; \
	  $(call run, ./ckpt_inflight +p4 +restart log ) || exit 1; \
	done
	rm -rf log
	$(call run, ./ckpt_inflight +p8 -direct )
	sync
	$(call run, ./ckpt_inflight +p4 +restart log )
	-rm -rf log

smptest: all

clean:
	rm -rf *.decl.h *.def.h *.o ckpt_inflight charmrun log
