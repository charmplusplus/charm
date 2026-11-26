-include ../../common.mk
-include ../../../include/conv-mach-opt.mak
CHARMC=../../../bin/charmc $(OPTS)

all: mcast_remote

mcast_remote.decl.h mcast_remote.def.h: mcast_remote.ci
	$(CHARMC) mcast_remote.ci

mcast_remote: mcast_remote.C mcast_remote.decl.h mcast_remote.def.h
	$(CHARMC) -language charm++ mcast_remote.C -o mcast_remote -module CkMulticast -module CommonLBs

test: mcast_remote
	$(call run, ./mcast_remote +p2 +balancer RotateLB +LBCommOn)
	$(call run, ./mcast_remote +p4 +balancer RotateLB +LBCommOn)

smptest: mcast_remote

clean:
	rm -f *.decl.h *.def.h *.o mcast_remote charmrun
