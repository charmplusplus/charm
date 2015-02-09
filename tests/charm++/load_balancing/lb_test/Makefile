-include ../../../common.mk
# Makefile for load-balancing test program

OPTS	= -O3
CHARMC	= ../../../../bin/charmc $(OPTS)
LBLIB	= -module CommonLBs

OBJS	= lb_test.o Topo.o

all:	lb_test lb_test.every

lb_test: $(OBJS)
	$(CHARMC) -language charm++ -o lb_test $(OBJS) $(LBLIB)

lb_test.every: $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS) -module EveryLB

lb_test.decl.h lb_test.def.h: lb_test.ci
	$(CHARMC) lb_test.ci 

Topo.decl.h Topo.def.h: Topo.ci
	$(CHARMC) Topo.ci 

lb_test.o: lb_test.C lb_test.decl.h Topo.decl.h
	$(CHARMC) -c lb_test.C

Topo.o: Topo.C Topo.h Topo.decl.h
	$(CHARMC) -c Topo.C

test:  lb_test lb_test.every
	$(call run, +p4 ./lb_test 100 100 10 40 10 1000 ring +balancer GreedyLB +LBDebug 1 )
	$(call run, +p4 ./lb_test 100 100 10 40 10 1000 ring +balancer CommLB +LBDebug 1 )
	$(call run, +p4 ./lb_test.every 100 100 10 40 10 1000 ring +balancer RecBipartLB +LBDebug 1 )

bgtest:  lb_test
	$(call run, +p4 ./lb_test 100 100 10 40 10 1000 ring +balancer CommLB +LBDebug 1 +x2 +y2 +z1 +cth1 +wth1 )

bgdump:
	$(call run, +p4 ./lb_test 100 100 10 40 10 1000 mesh2d +balancer DummyLB +LBDebug 1 +LBSameCpus +LBNoBackground +bgconfig ./bg_config +LBDump 1 +LBDumpFile lb_test.dump)

clean:
	rm -f *.decl.h *.def.h conv-host *.o lb_test charmrun lb_test.exe lb_test.pdb lb_test.ilk lb_test.every

