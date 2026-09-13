-include ../../../common.mk
CHARMC=../../../../bin/charmc -O3 $(OPTS)

OBJS = startup.o

all: startup

startup: $(OBJS)
	$(CHARMC) -language charm++ -module CommonLBs -g -o startup $(OBJS)

startup.decl.h: startup.ci
	$(CHARMC)  startup.ci

clean:
	rm -f *.decl.h *.def.h conv-host *.o startup charmrun *~

startup.o: startup.C startup.decl.h
	$(CHARMC) -c startup.C

test: all
	$(call run, ./startup +p4 200 20 +balancer GreedyLB +LBDebug 3 ++server ++server-port 1234)
