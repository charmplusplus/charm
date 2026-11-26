-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

# The Collide library is a module that the charm++ build target alone does not
# produce (`./build LIBS` or `all-test` does). Without it, this test is a no-op.
COLLIDE_LIB := $(wildcard ../../../lib/libmodulecollidecharm.a)

ifeq ($(COLLIDE_LIB),)
all test testp smptest:
	@echo "collide: skipped, libmodulecollidecharm.a is not in this build (built by ./build LIBS)"
else
all: collide_test

collide_test: collide_test.decl.h collide_test.C
	$(CHARMC) -language charm++ -module collidecharm collide_test.C -o collide_test

collide_test.decl.h: collide_test.ci
	$(CHARMC) collide_test.ci

# "2" runs fewer chunks than PEs, so some PEs have no contributor in a step.
test: collide_test
	$(call run, ./collide_test +p1)
	$(call run, ./collide_test +p4)
	$(call run, ./collide_test 2 +p4)
	$(call run, ./collide_test dist +p4)
	$(call run, ./collide_test 2 dist +p4)

testp: collide_test
	$(call run, ./collide_test +p$(P))
	$(call run, ./collide_test 1 +p$(P))
	$(call run, ./collide_test dist +p$(P))

smptest: collide_test
	$(call run, ./collide_test +p4 ++ppn 2)
	$(call run, ./collide_test 2 dist +p4 ++ppn 2)
endif

clean:
	rm -rf *.decl.h *.def.h *.o collide_test charmrun log
