-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

OBJS = ldbDemo.o ldbDemof.o

all: ldbDemo

ldbDemo: $(OBJS)
	$(CHARMC) -language f90charm -module CommonLBs -o ldbDemo $(OBJS)

ldbDemo.o: ldbDemo.C balanceDemo.decl.h
	$(CHARMC) -O -c ldbDemo.C

ldbDemof.o: ldbDemof.f90
	$(CHARMC) -c ldbDemof.f90

balanceDemo.decl.h: ldbDemo.ci
	$(CHARMC)   -language f90charm ldbDemo.ci 

test: ldbDemo
	$(call run, +p2 ./ldbDemo +balancer RotateLB +LBDebug 1)

clean:
	rm -f *.decl.h *.def.h conv-host *.o ldbDemo charmrun *.mod

