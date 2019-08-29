-include ../../../common.mk
CHARMC=../../../../bin/charmc $(OPTS)

OBJS = ldbRedDemo.o ldbRedDemof.o

all: ldbRedDemo

ldbRedDemo: $(OBJS)
	$(CHARMC) -language f90charm -module CommonLBs -o ldbRedDemo $(OBJS)

ldbRedDemo.o: ldbRedDemo.C balanceDemo.decl.h
	$(CHARMC) -O -c ldbRedDemo.C

ldbRedDemof.o: ldbRedDemof.f90
	$(CHARMC) -c ldbRedDemof.f90

balanceDemo.decl.h: ldbRedDemo.ci
	$(CHARMC) -language f90charm ldbRedDemo.ci

test: ldbRedDemo
	$(call run, +p2 ./ldbRedDemo +balancer RotateLB +LBDebug 1)

clean:
	rm -f *.decl.h *.def.h conv-host *.o ldbRedDemo charmrun *.mod
