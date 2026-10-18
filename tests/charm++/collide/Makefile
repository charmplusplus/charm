-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: collide_test

collide_test: collide_test.decl.h collide_test.C
	$(CHARMC) -language charm++ -module collidecharm collide_test.C -o collide_test

collide_test.decl.h: collide_test.ci
	$(CHARMC) collide_test.ci

test: collide_test
	$(call run, ./collide_test +p4)
	$(call run, ./collide_test +p4 +objid_tranche_log2 2)

smptest: test

clean:
	rm -rf *.decl.h *.def.h *.o collide_test charmrun
