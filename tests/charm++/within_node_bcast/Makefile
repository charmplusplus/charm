-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: within_node_bcast

within_node_bcast: within_node_bcast.decl.h within_node_bcast.def.h within_node_bcast.C
	$(CHARMC) -language charm++ within_node_bcast.C -o within_node_bcast

within_node_bcast.decl.h within_node_bcast.def.h: within_node_bcast.ci
	$(CHARMC) within_node_bcast.ci

clean:
	rm -f *.decl.h *.def.h *.o within_node_bcast charmrun

test: all
	$(call run, ./within_node_bcast +p1)
	$(call run, ./within_node_bcast +p2)

smptest:
	$(call run, ./within_node_bcast +p2 ++ppn 2)
	$(call run, ./within_node_bcast +p4 ++ppn 2)
