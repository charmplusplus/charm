-include ../../../common.mk
CHARMC	= ../../../../bin/charmc $(OPTS)

all: ping

ping: ping.decl.h ping.C
	$(CHARMC) ping.C -o ping -module CommonLBs

ping.decl.h: ping.ci
	$(CHARMC) ping.ci

test: ping
	$(call run, +p1 ./ping +balancer RotateLB +LBPeriod 0.1)
	$(call run, +p2 ./ping +balancer RotateLB +LBPeriod 0.1)

testp: ping
	$(call run, +p$(P) ./ping +balancer RotateLB +LBPeriod 0.1)

smptest: ping
	$(call run, +p2 ./ping +balancer RotateLB +LBPeriod 0.1 ++ppn 2)
	$(call run, +p4 ./ping +balancer RotateLB +LBPeriod 0.1 ++ppn 2)

clean:
	rm -rf *.decl.h *.def.h ping charmrun
