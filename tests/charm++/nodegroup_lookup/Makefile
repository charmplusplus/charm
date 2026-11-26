-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: nodegroup_lookup

nodegroup_lookup: nodegroup_lookup.decl.h nodegroup_lookup.def.h nodegroup_lookup.C
	$(CHARMC) -language charm++ nodegroup_lookup.C -o nodegroup_lookup

nodegroup_lookup.decl.h nodegroup_lookup.def.h: nodegroup_lookup.ci
	$(CHARMC) nodegroup_lookup.ci

clean:
	rm -f *.decl.h *.def.h *.o nodegroup_lookup charmrun

test: all
	$(call run, ./nodegroup_lookup +p1)
	$(call run, ./nodegroup_lookup +p4)

testp: all
	$(call run, ./nodegroup_lookup +p$(P))

smptest: all
	$(call run, ./nodegroup_lookup +p2 ++ppn 2)
	$(call run, ./nodegroup_lookup +p4 ++ppn 2)
