-include ../../../common.mk
# Driver for running LB strategies on a synthetic chare-level comm graph.

OPTS	= -O3
CHARMC	= ../../../../bin/charmc $(OPTS)
# DiffusionLB brings the CommonLBs module and METIS with it (its scratch-remap
# hand-off runs on a MetisLB it creates), so it must not be linked next to
# EveryLB, which carries a second copy of the same objects. +balancer MetisLB
# works from this alone.
LBLIB	= -module DiffusionLB

OBJS	= lbdriver.o

all:	lbdriver lbsim

lbdriver: $(OBJS)
	$(CHARMC) -language charm++ -o lbdriver $(OBJS) $(LBLIB)

lbdriver.decl.h lbdriver.def.h: lbdriver.ci
	$(CHARMC) lbdriver.ci

lbdriver.o: lbdriver.C lbdriver.decl.h
	$(CHARMC) -c lbdriver.C

# lbsim runs the same balancer's decision code on N virtual nodes inside one
# process; see the header of lbsim.C. It needs the module's own headers, which
# are not installed, and METIS's.
SIMINC	= -I../../../../src/ck-ldb -I../../../../src/libs/ck-libs/metis/libmetis

lbsim: lbsim.o
	$(CHARMC) -language charm++ -o lbsim lbsim.o $(LBLIB)

lbsim.decl.h lbsim.def.h: lbsim.ci
	$(CHARMC) lbsim.ci

lbsim.o: lbsim.C lbsim.decl.h
	$(CHARMC) $(SIMINC) -c lbsim.C

# 128x128 objects on 256 virtual nodes, four diffusion steps. The default
# neighbour count is 1, which the two ring edges already exceed, so the graph
# would be the ring alone; 6 lets every node add its four stencil partners.
sim: lbsim
	$(call run, +p1 ./lbsim 128 128 256 4 +LBDiffusionCommOn +LBDiffusionNumNbors 6)
	python3 plot_map.py lbsim.json lbsim.html

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
	rm -f *.decl.h *.def.h conv-host *.o lbdriver lbsim charmrun lbdriver.json lbsim.json
