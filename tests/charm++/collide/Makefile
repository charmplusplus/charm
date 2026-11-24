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

test: collide_test
	$(call run, ./collide_test +p4)
	$(call run, ./collide_test +p4 +objid_tranche_log2 2)

smptest: test
endif

clean:
	rm -rf *.decl.h *.def.h *.o collide_test charmrun
