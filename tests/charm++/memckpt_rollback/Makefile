-include ../../common.mk
CHARMC = ../../../bin/charmc $(OPTS)

all: memckpt_rollback

memckpt_rollback: memckpt_rollback.o
	$(CHARMC) -language charm++ -o $@ memckpt_rollback.o -module CommonLBs

memckpt_rollback.decl.h: memckpt_rollback.ci
	$(CHARMC) memckpt_rollback.ci

memckpt_rollback.o: memckpt_rollback.C memckpt_rollback.decl.h
	$(CHARMC) -c memckpt_rollback.C

test: all testrestriction
	$(call run, ./memckpt_rollback +p1 )
	$(call run, ./memckpt_rollback +p2 )
	$(call run, ./memckpt_rollback +p4 )
	$(call run, ./memckpt_rollback +p4 -rollbacks 2 )
	$(call run, ./memckpt_rollback +p4 -lbfirst +balancer RotateLB )

# A migration, an insertion or a deletion between the checkpoint and the
# rollback must abort the rollback.
# $(1): program options, $(2): the word in the refusal message
define refusal
	@$(call run, ./memckpt_rollback +p4 $(1) ) > restriction.out 2>&1; rc=$$?; \
	if [ $$rc -eq 0 ]; then \
	  cat restriction.out; echo "testrestriction ($(1)): expected a nonzero exit"; exit 1; \
	elif ! grep -q "rollback refused: .* array elements $(2)" restriction.out; then \
	  cat restriction.out; echo "testrestriction ($(1)): restriction message missing"; exit 1; \
	else \
	  echo "memckpt_rollback testrestriction ($(1)) passed"; \
	fi
endef

testrestriction: all
	$(call refusal,-lb +balancer RotateLB,migrated)
	$(call refusal,-insert,inserted)
	$(call refusal,-delete,deleted)

testp: all
	$(call run, ./memckpt_rollback +p$(P) )

smptest: all
	$(call run, ./memckpt_rollback +p4 ++ppn 2 )

clean:
	rm -f *.decl.h *.def.h *.o memckpt_rollback charmrun restriction.out
