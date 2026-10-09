-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: objid_bound

# RotateLB is built into the core; GreedyLB comes from CommonLBs.
objid_bound: objid_bound.decl.h objid_bound.C
	$(CHARMC) objid_bound.C -o objid_bound -module CommonLBs

objid_bound.decl.h: objid_bound.ci
	$(CHARMC) objid_bound.ci

test: objid_bound
	$(call run, ./objid_bound +p4)
	$(call run, ./objid_bound +p4 -lb +balancer RotateLB)
	$(call run, ./objid_bound +p4 -lb +balancer GreedyLB)

smptest: objid_bound
	$(call run, ./objid_bound +p4)
	$(call run, ./objid_bound +p4 -lb +balancer RotateLB)
	$(call run, ./objid_bound +p4 -lb +balancer GreedyLB)

clean:
	rm -rf *.decl.h *.def.h objid_bound charmrun
