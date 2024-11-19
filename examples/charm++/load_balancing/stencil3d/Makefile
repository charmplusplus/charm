-include ../../../common.mk
OPTS	= -O3
#CHARMC	=  ../../../../netlrts-linux-x86_64/bin/charmc $(OPTS)
CHARMC= ../../../../bin/charmc $(OPTS)
##ofi-linux-x86_64-cxi-slurmpmi2cray-smp/bin/charmc $(OPTS)

OBJS = stencil3d.o

all: stencil3d

stencil3d: $(OBJS)
	$(CHARMC) -language charm++ -module CommonLBs -o stencil3d $(OBJS)

projections: $(OBJS)
	$(CHARMC) -language charm++ -module CommonLBs -tracemode projections -lz -o stencil3d.prj $(OBJS)

summary: $(OBJS)
	$(CHARMC) -language charm++ -module CommonLBs -tracemode summary -lz -o stencil3d.sum $(OBJS)

stencil3d.decl.h: stencil3d.ci
	$(CHARMC)  stencil3d.ci

stencil3d.o: stencil3d.C stencil3d.decl.h
	$(CHARMC) -c stencil3d.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o stencil3d stencil3d.prj charmrun *~

test: stencil3d
	$(call run, +p4 ./stencil3d 64 32 +balancer RefineLB +LBDebug 1 )
	$(call run, +p4 ./stencil3d 64 32 +balancer GreedyLB +LBDebug 1 )
	$(call run, +p4 ./stencil3d 64 32 +balancer GreedyRefineLB +LBDebug 1 )
	$(call run, +p4 ./stencil3d 64 32 +balancer DistributedLB +LBDebug 1 )
	$(call run, +p4 ./stencil3d 64 32 +balancer MetisLB +LBDebug 1 )


testp: stencil3d
	$(call run, +p$(P) ./stencil3d $$(( $(P) * 8 )) 4 +balancer RefineLB +LBDebug 1 )
	$(call run, +p$(P) ./stencil3d $$(( $(P) * 8 )) 4 +balancer GreedyLB +LBDebug 1 )
	$(call run, +p$(P) ./stencil3d $$(( $(P) * 8 )) 4 +balancer GreedyRefineLB +LBDebug 1 )
	$(call run, +p$(P) ./stencil3d $$(( $(P) * 8 )) 4 +balancer DistributedLB +LBDebug 1 )
	$(call run, +p$(P) ./stencil3d $$(( $(P) * 8 )) 4 +balancer MetisLB +LBDebug 1 )
