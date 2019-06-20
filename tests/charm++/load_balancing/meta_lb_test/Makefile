-include ../../../common.mk
CHARMC	= ../../../../bin/charmc $(OPTS)

all: period_selection

period_selection: period_selection.decl.h period_selection.C
	$(CHARMC) period_selection.C -o period_selection -module RotateLB

period_selection.decl.h: period_selection.ci
	$(CHARMC) period_selection.ci

test: period_selection
	$(call run, +p1 ./period_selection +balancer RotateLB +LBPeriod 0.001 +MetaLB)
	$(call run, +p2 ./period_selection +balancer RotateLB +LBPeriod 0.001 +MetaLB)
	$(call run, +p4 ./period_selection +balancer RotateLB +LBPeriod 0.001 +MetaLB)
ifeq ($(CMK_SMP),1)
	$(call run, +p2 ./period_selection +balancer RotateLB +LBPeriod 0.001 +MetaLB ++ppn 2)
	$(call run, +p4 ./period_selection +balancer RotateLB +LBPeriod 0.001 +MetaLB ++ppn 2)
endif

clean:
	rm -rf *.decl.h *.def.h period_selection charmrun

