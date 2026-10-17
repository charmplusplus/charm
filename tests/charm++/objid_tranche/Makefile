-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: objid_tranche

objid_tranche: objid_tranche.decl.h objid_tranche.C
	$(CHARMC) objid_tranche.C -o objid_tranche

objid_tranche.decl.h: objid_tranche.ci
	$(CHARMC) objid_tranche.ci

# 3 unbounded arrays plus demand creation ([createhome]) on each, with default
# tranches (no refill), 64-id tranches (refills), 4-id tranches (deferral).
test: objid_tranche
	$(call run, ./objid_tranche +p4 )
	$(call run, ./objid_tranche +p4 +objid_tranche_log2 6 )
	$(call run, ./objid_tranche +p4 +objid_tranche_log2 2 )

smptest: test

clean:
	rm -f *.decl.h *.def.h *.o objid_tranche charmrun
