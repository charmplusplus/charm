-include ../../../common.mk
OPTS	= -O3
CHARMC	= ../../../../bin/charmc $(OPTS)

OBJS	= kNeighbor.o

all: kNeighbor 

kNeighbor: $(OBJS)
	$(CHARMC) -language charm++ -module CommonLBs -o kNeighbor $(OBJS)

projections: $(COMPOBJS)
	$(CHARMC) -language charm++ -tracemode projections -lz -o kNeighbor.prj $(OBJS)

kNeighbor.decl.h: kNeighbor.ci
	$(CHARMC) kNeighbor.ci

kNeighbor.o: kNeighbor.C kNeighbor.decl.h
	$(CHARMC) -c kNeighbor.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o charmrun *~ kNeighbor kNeighbor.prj

test: all
	$(call run, +p4 ./kNeighbor 4 50 128 10 +balancer RefineLB +LBDebug 1 )
	$(call run, +p4 ./kNeighbor 4 50 128 10 +balancer GreedyLB +LBDebug 1 )
	$(call run, +p4 ./kNeighbor 4 50 128 10 +balancer GreedyRefineLB +LBDebug 1 )
	$(call run, +p4 ./kNeighbor 4 50 128 10 +balancer DistributedLB +LBDebug 1 )
	$(call run, +p4 ./kNeighbor 4 50 128 10 +balancer MetisLB +LBDebug 1 )

testp: all
	$(call run, +p$(P) ./kNeighbor $$(( $(P) * 8 )) 50 128 10 +balancer RefineLB +LBDebug 1 )
	$(call run, +p$(P) ./kNeighbor $$(( $(P) * 8 )) 50 128 10 +balancer GreedyLB +LBDebug 1 )
	$(call run, +p$(P) ./kNeighbor $$(( $(P) * 8 )) 50 128 10 +balancer GreedyRefineLB +LBDebug 1 )
	$(call run, +p$(P) ./kNeighbor $$(( $(P) * 8 )) 50 128 10 +balancer DistributedLB +LBDebug 1 )
	$(call run, +p$(P) ./kNeighbor $$(( $(P) * 8 )) 50 128 10 +balancer MetisLB +LBDebug 1 )
