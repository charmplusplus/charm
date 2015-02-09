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
	$(call run, +p4 ./kNeighbor 4 50 128 10 +balancer GreedyLB +LBDebug 1 )

bgtest: all
	$(call run, +p4 ./kNeighbor 4 50 128 10 +balancer GreedyLB +x2 +y2 +z1 +cth1 +wth1 +LBDebug 1 )

