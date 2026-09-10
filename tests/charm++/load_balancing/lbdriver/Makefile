-include ../../../common.mk
# Driver for running LB strategies on a synthetic chare-level comm graph.

OPTS	= -O3
CHARMC	= ../../../../bin/charmc $(OPTS)
# DiffusionLB is not part of EveryLB, so both modules are named explicitly.
LBLIB	= -module EveryLB -module DiffusionLB

OBJS	= lbdriver.o

all:	lbdriver

lbdriver: $(OBJS)
	$(CHARMC) -language charm++ -o lbdriver $(OBJS) $(LBLIB)

lbdriver.decl.h lbdriver.def.h: lbdriver.ci
	$(CHARMC) lbdriver.ci

lbdriver.o: lbdriver.C lbdriver.decl.h
	$(CHARMC) -c lbdriver.C

# 32x32 objects on 4 PEs, each its own logical node. A Charm node is a process,
# so on this single-process build numNodes would be 1 and DiffusionLB's
# across-node phase -- the diffusion itself -- would never run;
# CHARM_DIFFUSION_NODE_SIZE=1 splits the process into 4 one-PE nodes instead.
# Within-node balancing is then a no-op, so every change in the third plot comes
# from across-node diffusion alone.
test: lbdriver
	CHARM_DIFFUSION_NODE_SIZE=1 $(call run, +p4 ./lbdriver 32 32 6 4096 MetisLB DiffusionLB \
	  +balancer MetisLB +balancer DiffusionLB +LBCommOn +LBDiffusionCommOn)
	python3 plot_map.py lbdriver.json lbmap.html

clean:
	rm -f *.decl.h *.def.h conv-host *.o lbdriver charmrun lbdriver.json
