-include ../../common.mk
CHARMC=../../../bin/charmc $(OPTS)

all: objid_wide

# GreedyLB comes from CommonLBs.
objid_wide: objid_wide.decl.h objid_wide.C
	$(CHARMC) objid_wide.C -o objid_wide -module CommonLBs

objid_wide.decl.h: objid_wide.ci
	$(CHARMC) objid_wide.ci

# The abort runs exit non-zero, which the pipe hides; grep checks the text.
test: objid_wide
	$(call run, ./objid_wide +p4 )
	$(call run, ./objid_wide +p4 -wide 2000 -lb +balancer GreedyLB )
	-rm -rf ckptlog
	$(call run, ./objid_wide +p4 -wide 3000 -ckpt )
	$(call run, ./objid_wide +p4 +restart ckptlog )
	$(call run, ./objid_wide +p2 +restart ckptlog )
	-rm -rf ckptlog
	./objid_wide +p4 -abort nobounds 2>&1 | grep -q "but no bounds"
	./objid_wide +p4 -abort bigbounds 2>&1 | grep -q "bits to pack its index, but a wide-collection"
	./objid_wide +p4 -wide 10 -standard 2100 2>&1 | grep -q "chare arrays created, half of the"
	./objid_wide +p4 -wide 10 -standard 4200 2>&1 | grep -q "ran out of collection numbers"

smptest: test

clean:
	rm -f *.decl.h *.def.h *.o objid_wide charmrun
	rm -rf ckptlog/
