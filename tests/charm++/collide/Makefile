-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

# The Collide library is a module that the charm++ build target alone does not
# produce (`./build LIBS` does). Without it, this test is a no-op.
COLLIDE_LIB := $(wildcard ../../../lib/libmodulecollidecharm.a)

ifeq ($(COLLIDE_LIB),)
all test smptest:
	@echo "collide: skipped, libmodulecollidecharm.a is not in this build (built by ./build LIBS)"
else
all: collide_test

collide_test: collide_test.decl.h collide_test.C
	$(CHARMC) -language charm++ -module collidecharm collide_test.C -o collide_test

collide_test.decl.h: collide_test.ci
	$(CHARMC) collide_test.ci

# "2" runs fewer chunks than PEs, so some PEs have no contributor in a step.
test: collide_test
	$(call run, ./collide_test +p4)
	$(call run, ./collide_test +p4 +objid_tranche_log2 2)
	$(call run, ./collide_test 2 +p4)
	$(call run, ./collide_test dist +p4)
	$(call run, ./collide_test 2 dist +p4)
	# Checkpoint after step 2, then restart on the same and on fewer PEs.
	-rm -rf log
	$(call run, ./collide_test ckpt +p4)
	-sync
	$(call run, ./collide_test +p4 +restart log)
	$(call run, ./collide_test +p2 +restart log)
	-rm -rf log
	$(call run, ./collide_test dist ckpt +p4)
	-sync
	$(call run, ./collide_test +p3 +restart log)
	-rm -rf log

smptest: test
endif

clean:
	rm -rf *.decl.h *.def.h *.o collide_test charmrun log
