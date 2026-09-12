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

# The standalone lbsim: the same simulation without the Charm++ runtime, so
# it runs on a login node (the runtime's startup wants a GPU and the NIC).
# Compiled against the same headers, linked with METIS, the decision code,
# and lbsim_stub.C, which supplies the few runtime symbols the decision code
# touches. Takes the same arguments and +LB flags. See the note at the top
# of lbsim.C.
SA_CXX	= g++ -std=c++17 -O2
CUDA_DIR := $(shell sed -n 's/^CUDA_DIR="\(.*\)"/\1/p' ../../../../include/conv-mach-opt.sh)
SA_INC	= -I../../../../include $(SIMINC) -I$(CUDA_DIR)/include
SA_SRC	= lbsim.C lbsim_stub.C ../../../../src/ck-ldb/DiffusionMetric.C
SA_DEPS	= ../../../../src/ck-ldb/DiffusionSelect.h ../../../../src/ck-ldb/DiffusionFlow.h \
	  ../../../../src/ck-ldb/DiffusionMetric.h ../../../../src/ck-ldb/DiffusionLoad.h \
	  ../../../../src/ck-ldb/LBLoadDim.h ../../../../src/ck-ldb/DiffusionCostModel.h \
	  ../../../../src/ck-ldb/DiffusionCostModel.C

lbsim-standalone: $(SA_SRC) $(SA_DEPS)
	$(SA_CXX) -DLBSIM_STANDALONE $(SA_INC) -o $@ $(SA_SRC) ../../../../lib/libckmetis.a

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
	rm -f *.decl.h *.def.h conv-host *.o lbdriver lbsim lbsim-standalone charmrun lbdriver.json lbsim.json
